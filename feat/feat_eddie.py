import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pickle
import logging
import sys
import json
import os
import random
from feat.feat_plan import encode_plan_index_config, PlanTreeEncoder
from util.string_util import string_to_md5


def eddie_feat_data(all_samples, db_stats, max_sample_cnt=None, enable_histogram=True):
    feat_params = {"max_index_col_num": 3, "max_group_col_num": 5, "max_sort_col_num": 5, "enable_histogram": enable_histogram}
    encoder = PlanTreeEncoder(db_stats, feat_params)

    feat_dicts = []

    len_samples = len(all_samples)
    logging.info(f"Input samples len: {len_samples}")
    
    if max_sample_cnt and len_samples > max_sample_cnt:
        logging.info("samples len exceed max cnt, do random sampling")
        all_samples = random.sample(all_samples, max_sample_cnt)
    
    discard_sample_cnt = 0
    error_samples = []
    for si, sample in enumerate(all_samples):
        if (si+1) % 500 == 0:
            logging.info(f"encoded {si+1}/{len_samples}")
        query = sample["query"]
        init_index_config = sample["init_indexes"]
        index_config = sample["indexes_str"]
        orig_plan = sample["plan_tree"]
        orig_times = sample["orig_times"]
        with_index_runtimes = sample["with_index_runtimes"]
        with_index_plan = sample["with_index_plan"]
        query_type = sample["query_type"]
        if orig_times is None:
            discard_sample_cnt += 1 # If the initial index times out, the label cannot be calculated and the sample is discarded
            logging.info(f"sample_id {si} not compute label, droped sample {sample}")
            continue
        if with_index_runtimes is None:
            with_index_runtimes, with_index_plan = orig_times, orig_plan
        
        if orig_times > 0.0 and orig_times >= with_index_runtimes:
            label = (orig_times - with_index_runtimes) / orig_times
        else:
            label = 0
            
        feat_dict = encode_plan_index_config(encoder, orig_plan, index_config, label)

        if 'heights' not in feat_dict['struct']:
            feat_dict['struct']["heights"] = []
            continue
        if "shortest_path_mat" not in feat_dict["struct"]:
            feat_dict["struct"]['shortest_path_mat'] = []
        if "filter_dict" not in feat_dict["struct"]:
            feat_dict["struct"]['filter_dict'] = dict()
        if "join_dict" not in feat_dict["struct"]:
            feat_dict["struct"]['join_dict'] = dict()
        if "index_cond_dict" not in feat_dict["struct"]:
            feat_dict["struct"]['index_cond_dict'] = dict()
        feat_dict["sql_md5"] = string_to_md5(query.text)
        feat_dict["sample_id"] = si
        feat_dict["query_type"] = query_type
        feat_dict["query"] = query
        feat_dict["sql"] = query.text
        feat_dict["orig_plan"] = orig_plan
        feat_dict["indexes_str"] = index_config
        feat_dict["plan_tree"] = sample["plan_tree"]
            
        feat_dicts.append(feat_dict)
    return feat_dicts
