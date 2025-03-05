import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from selection.dbms.postgres_dbms import PostgresDatabaseConnector
from selection.algorithms.auto_admin_algorithm import AutoAdminAlgorithm
from selection.algorithms.anytime_algorithm import AnytimeAlgorithm
from selection.end2end_utils import *

from selection.workload import Workload
import logging
import pickle
import itertools
import random
import torch
from run_lib import Args

import time
from data_process.data_process import *
from util import random_util
from selection import cost_evaluation_lib
from selection import end2end_whatif

from lib_est.lib_model import *
from lib_est.lib_dataset import *


logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')


def calculate_indexes(workload, algorithm_type, db_connector, db_stats, model, max_indexes=3):
    logging.info(f"use algorithm_type: {algorithm_type}")
    st = time.time()
    cost_evaluation = cost_evaluation_lib.CostEvaluation(db_connector = db_connector, db_stat=db_stats, model=model)
    if algorithm_type == DTA:
        dta_parameters = {'max_index_width': 2, 'budget_MB': 2000, 'max_runtime_minutes': 10, 'benchmark_name': 'example', "CostEvaluation": cost_evaluation}
        algorithm = AnytimeAlgorithm(db_connector, dta_parameters)  # dta
    elif algorithm_type == AUTO_ADMIN:
        auto_admin_parameters = { "max_indexes": max_indexes, "max_indexes_naive": 1, "max_index_width": 2, "CostEvaluation": cost_evaluation}
        algorithm = AutoAdminAlgorithm(db_connector, auto_admin_parameters)
    
    indexes = algorithm.calculate_best_indexes(workload)

    ind_cfg = []
    for ind in indexes:
        ind_str = ind.table().name + "#" + ",".join([c.name for c in ind.columns])
        ind_cfg.append(ind_str)
    ind_cfg.sort()
    logging.info(f"calculate_indexes ratio: {algorithm.cost_evaluation.cache_hits} / {algorithm.cost_evaluation.cost_requests} = {algorithm.cost_evaluation.cache_hits / algorithm.cost_evaluation.cost_requests}")
    logging.info(f"lib calculate_indexes: {ind_cfg}")
    logging.info(f"calculate_indexes runtime: {time.time() - st} s")
    return ind_cfg



def run_main(dataset_path, algorithm_type= AUTO_ADMIN, db_name = "indexselection_tpcds___10", db_stats=None, model=None):
    random_util.seed_everything(0)
    
    db_connector = PostgresDatabaseConnector(db_name, autocommit=True)
    db_connector.drop_indexes()
    db_connector.commit()
    
    test_workload = get_test_workload(dataset_path, k_flod=10)
    test_workload = Workload(test_workload.queries[:1])
    orig_total_time, query2runtime = get_workload_runtime(test_workload, db_connector)
    index_config = calculate_indexes(test_workload, algorithm_type, db_connector, db_stats, model)
    total_runtime_with_indexes, regression_query = evaluate_indexes_quality(test_workload, index_config, query2runtime, db_connector)
    logging.info("LIB")
    logging.info(f"{algorithm_type} (ms) orig_total_time time: {orig_total_time}")
    logging.info(f"{algorithm_type} (ms) total_runtime_with_indexes: {total_runtime_with_indexes}")
    logging.info(f"{algorithm_type} cost saving: (ms) {orig_total_time - total_runtime_with_indexes}")
    logging.info(f"{algorithm_type} cost saving rate: {(orig_total_time - total_runtime_with_indexes) / orig_total_time}")
    logging.info(f"regression_query: {regression_query}")
    logging.info(f"{algorithm_type} total_runtime_with_indexes: {total_runtime_with_indexes/1000} s  orig_total_time time: {orig_total_time/1000} s")
    
    # eval_what_if
    end2end_whatif.eval_what_if(test_workload, algorithm_type, db_connector, query2runtime, orig_total_time)
    
    return total_runtime_with_indexes, regression_query


def eval_lib(test_workload, algorithm_type, db_connector, query2runtime, orig_total_time, data_path, db_stats, selected_fold=0, max_indexes=3, times=5):
    logging.info("Strat LIB indexes_quality")
        
    fn = data_path.split("/")[-1].split(".")[0]
    args = Args()
    
    # lib
    encoder_model, pooling_model = make_model(args.dim1, args.n_encoder_layers, args.dim3, args.n_heads, dropout=args.dropout_r)
    model = self_attn_model(encoder_model, pooling_model, 12, args.dim1, args.dim2)
    
    model_save_path = f"./checkpoints/{fn}_lib_v5_fold_{selected_fold}.pth"
    checkpoint = torch.load(model_save_path, map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    #
    
    index_config = calculate_indexes(test_workload, algorithm_type, db_connector, db_stats, model, max_indexes=max_indexes)
    total_runtime_with_indexes, regression_query = evaluate_indexes_quality(test_workload, index_config, query2runtime, db_connector, times=times)
    logging.info("LIB")
    logging.info(f"{algorithm_type} (ms) orig_total_time time: {orig_total_time}")
    logging.info(f"{algorithm_type} (ms) total_runtime_with_indexes: {total_runtime_with_indexes}")
    logging.info(f"{algorithm_type} cost saving: (ms) {orig_total_time - total_runtime_with_indexes}")
    logging.info(f"{algorithm_type} cost saving rate: {(orig_total_time - total_runtime_with_indexes) / orig_total_time}")
    logging.info(f"regression_query: {regression_query}")
    logging.info(f"{algorithm_type} total_runtime_with_indexes: {total_runtime_with_indexes/1000} s  orig_total_time time: {orig_total_time/1000} s")
    logging.info("End LIB indexes_quality")
    return {"runtime_with_indexes(ms)": total_runtime_with_indexes, "regression_querys":regression_query, "regression": len(regression_query), "cost_saving(ms)": orig_total_time - total_runtime_with_indexes, "improve": (orig_total_time - total_runtime_with_indexes) / orig_total_time, "orig_runtime(ms)":orig_total_time }