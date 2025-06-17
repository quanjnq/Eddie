import random
import logging
import math
import os
import pickle
import json
import copy

from util.string_util import string_to_md5


def data_preprocess(dataset_path, db_stat_path, log_label=False, random_change_tbl_col=False):
    with open(dataset_path, "rb") as f:
        src_data = pickle.load(f)
    with open(db_stat_path, "r") as f:
        db_stat = json.load(f)
        
    res_data = []
    
    for si, sample in enumerate(src_data):
        query = sample[0]
        init_index_comb, index_comb = sample[1]
        orig_times, orig_plan = sample[2]
        with_index_runtimes, with_index_plan = sample[3]
        if len(sample) >= 5:
            query_type = sample[4]
        elif len(init_index_comb) > 0:
            query_type = "init_index"
        else:
            query_type = "unk"
        if with_index_runtimes is None:
            with_index_runtimes, with_index_plan = orig_times, orig_plan
        # If the initial index encounters a timeout, set the label of the timeout sample to 0
        if orig_times > 0.0 and orig_times >= with_index_runtimes:
            label = (orig_times - with_index_runtimes) / orig_times
        else:
            label = 0
        if log_label:
            label = math.log(1 + label)
            
        sql = query.text
        sql_md5 = string_to_md5(sql)
        
        indexes = []
        for index_str in index_comb:
            tbl = index_str.split("#")[0]
            cols = index_str.split("#")[1].split(",")
            indexes.append([f"{tbl}.{col}" for col in cols])
        
        data_item = {
            "sql": sql,
            "sample_id": si,
            "plan_tree": orig_plan,
            "with_index_plan": with_index_plan,
            "orig_plan": orig_plan,
            "orig_times": orig_times,
            "with_index_runtimes": with_index_runtimes,
            "init_indexes": init_index_comb,
            "indexes_str": index_comb,
            "indexes": indexes,
            "label": label,
            "sql_md5": sql_md5,
            "query_type": query_type,
            "query": query,
            "split_key": f"{query_type}_{query.nr}"
        }
        res_data.append(data_item)
    
    if random_change_tbl_col:
        res_data = random_change_schema(res_data, db_stat)
    
    return res_data, db_stat


def split_key_of(data_item, enable_template_split=False):
    if enable_template_split:
        split_key = str(data_item['query'].nr)+"#"+data_item['query_type']
    else:
        split_key = data_item['sql_md5']
    return split_key


def split_dataset_by_sql_kfold(data_items, k_folds, col_change_mapping=None, enable_template_split=False):
    # Group data items by their `sql_md5` key
    sql_2_samples = {}
    for data in data_items:
        split_key = split_key_of(data, enable_template_split)
        
        if split_key not in sql_2_samples:
            sql_2_samples[split_key] = []
        sql_2_samples[split_key].append(data)

    # Shuffle the keys for randomness
    sql_keys = list(sql_2_samples.keys())
    # sql_keys.sort()
    random.shuffle(sql_keys)
    
    # Divide the keys into k folds, distributing extra items evenly
    fold_size = len(sql_keys) // k_folds
    remainder = len(sql_keys) % k_folds
    
    # Create folds, ensuring extra items are evenly distributed
    folds = []
    start_idx = 0
    for i in range(k_folds):
        end_idx = start_idx + fold_size + (1 if i < remainder else 0)
        folds.append(sql_keys[start_idx:end_idx])
        start_idx = end_idx
    logging.info(f"sql count: {len(sql_keys)}, divide into {k_folds} folds: {[len(fold) for fold in folds]}")
    
    # Generate train and validation sets for each fold
    datasets = []
    for i in range(k_folds):
        # Current fold is used as the validation set
        val_keys = folds[i]
        # All other folds are combined to form the training set
        train_keys = [key for j, fold in enumerate(folds) if j != i for key in fold]

        # Retrieve the actual data items based on the keys
        train_items = [item for item in data_items if split_key_of(item, enable_template_split) in train_keys]
        val_items = [item for item in data_items if split_key_of(item, enable_template_split) in val_keys]
        
        if col_change_mapping:
            val_items = copy.deepcopy(val_items)
            for item in val_items:
                plantree_str = json.dumps(item["plan_tree"])
                
                for old_name, new_name in col_change_mapping.items():
                    plantree_str = plantree_str.replace(old_name, new_name)
                    
                    for ind in item["indexes"]:
                        for i in range(len(ind)):
                            ind[i] = ind[i].replace(old_name, new_name)
                            
                item["plan_tree"] = json.loads(plantree_str)
                
        # Append the (train, val) pair to the result
        datasets.append((train_items, val_items))

    return datasets


def random_change_schema(data_items, db_stat):
    col_names = set()
    for tbl_col in db_stat.keys():
        tbl, col = tbl_col.split('.')
        col_names.add(col)
    col_change_mapping = random_change_col_name(col_names, rate=0.2)

    for item in data_items:
        plantree_str = json.dumps(item["plan_tree"])
        
        for old_name, new_name in col_change_mapping.items():
            plantree_str = plantree_str.replace(old_name, new_name)
            
            for ind in item["indexes"]:
                for i in range(len(ind)):
                    ind[i] = ind[i].replace(old_name, new_name)
                    
        item["plan_tree"] = json.loads(plantree_str)
    
    # Update column names in db_stat that have been changed
    new_db_stat = {}
    for tbl_col, stat in db_stat.items():
        tbl, col = tbl_col.split('.')
        if col in col_change_mapping:
            new_col = col_change_mapping[col]
            new_tbl_col = f"{tbl}.{new_col}"
            new_db_stat[new_tbl_col] = stat
        else:
            new_db_stat[tbl_col] = stat
    db_stat.clear()
    db_stat.update(new_db_stat)

    return data_items


def random_change_col_name(col_names, rate=0.2):
    saved_seed = random.getstate()
    col_name_change_mapping = {col: f"{col}_1" for col in random.sample(list(col_names), int(rate * len(col_names)))}
    random.setstate(saved_seed)

    return col_name_change_mapping
