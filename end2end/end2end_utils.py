import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from selection.dbms.postgres_dbms import PostgresDatabaseConnector
from selection.algorithms.auto_admin_algorithm import AutoAdminAlgorithm
from selection.algorithms.anytime_algorithm import AnytimeAlgorithm

from selection.workload import Workload
import logging
import pickle
import itertools
import time
from data_process.data_process import *

DTA = "DTA"
AUTO_ADMIN = "AutoAdmin"

def get_avg_runtimes(query, timeout=120*1000, db_connector=None):
    each_times = []    
    times = 4
    for k in range(times): 
        actual_runtimes, plan = db_connector.exec_query(query, timeout=timeout)
        if actual_runtimes is None and k > 0:
            return None, None
        each_times.append(actual_runtimes)
        
    if times > 1:
        avg_times = sum(each_times[1:]) / len(each_times[1:])
    else:
        avg_times = each_times[0]
    return avg_times, plan
    
    
def clear_all_indexes(db_connector):
    stmt = "SELECT * FROM hypopg_reset();"
    db_connector.exec_only(stmt)
    stmt = "select indexname from pg_indexes where schemaname='public'"
    indexes = db_connector.exec_fetch(stmt, one=False)
    for index in indexes:
        index_name = index[0]
        drop_stmt = "drop index {}".format(index_name)
        db_connector.exec_only(drop_stmt)
        db_connector.commit()
        
        
def create_indexes(db_connector, indexes):
    for index in indexes:
        sp = index.split("#")
        tbl, cols = sp[0], sp[1]
        statement = f"create index on {tbl} ({cols})"
        db_connector.exec_only(statement)
        db_connector.commit()
        print(statement)


def evaluate_indexes_quality(workload, index_config, query2runtime, db_connector, times=5):
    total_runtime_with_indexes = 0
    regression_query = []
    
    # create index
    create_indexes(db_connector, index_config)
    sql2plan = {}
    logging.info(f"timeout by {times} times")
    for query in workload.queries:
        orig_runtimes = query2runtime[query]
        
        query_timeout = max(int(orig_runtimes * times), 1)
        
        avg_times, plan = get_avg_runtimes(query, query_timeout, db_connector=db_connector)
        
        if avg_times is None or orig_runtimes < avg_times:
            regression_query.append(query)
            total_runtime_with_indexes += query_timeout if avg_times is None else avg_times
        else:
            total_runtime_with_indexes += avg_times
        logging.info(f"{(query, 'orig:'+ str(orig_runtimes), 'with_ind:'+str(avg_times))}")
    
    clear_all_indexes(db_connector)
    
    return total_runtime_with_indexes, regression_query


def get_test_workload(dataset_path, k_flod=10, selected_fold=1, rs=-1):
    logging.info(f"loading dataset from {dataset_path}")
    logging.info(f"selected test fold {selected_fold} / {k_flod}")
    with open(dataset_path, "rb") as f:
        src_data = pickle.load(f)
    data_items = data_preprocess(src_data)
    if rs > 0:
        cs = random.getstate()
        random.seed(rs)
        random.shuffle(data_items)
        random.setstate(cs)
    
    # k-fold cross validation
    k_folds_datasets = split_dataset_by_sql_kfold(data_items, k_flod)
    
    val_items = k_folds_datasets[selected_fold][1]
    sql_set = set()
    test_items = []
    for item in val_items:
        if item["sql"] not in sql_set:
            sql_set.add(item["sql"])
            test_items.append(item)
    
    queries = [item["query"] for item in test_items]
    workload = Workload(queries)
    logging.info(f"test workload size {len(queries)}")
    return workload



def get_all_test_workload(dataset_path, k_flod=10, selected_fold=1):
    logging.info(f"loading dataset from {dataset_path}")
    with open(dataset_path, "rb") as f:
        src_data = pickle.load(f)
    data_items = data_preprocess(src_data)
    
    val_items = data_items
    sql_set = set()
    test_items = []
    for item in val_items:
        if item["sql"] not in sql_set:
            sql_set.add(item["sql"])
            test_items.append(item)
    # val_items = val_items[:1]
    
    queries = [item["query"] for item in test_items]
    workload = Workload(queries)
    logging.info(f"test workload size {len(queries)}")
    return workload

def split_into_groups(lst, group_size=10):
    return [lst[i:i + group_size] for i in range(0, len(lst), group_size)]

def get_workload_group(workload_path, w_size=10):
    logging.info(f"loading dataset from {workload_path}")
    with open(workload_path, "rb") as f:
        w = pickle.load(f)
    query_gps = split_into_groups(w.queries, w_size)
    print("group workload size:", [len(qs) for qs in  query_gps])
    ws = [Workload(qs) for qs in  query_gps]
    return ws

def get_workload_runtime(workload, db_connector):
    query2runtime = {}
    orig_total_time = 0
    sql2plan = {}
    print()
    for i, query in enumerate(workload.queries):
        query_orig_time,plan = get_avg_runtimes(query,db_connector=db_connector)
        sql2plan[query.text] = (query_orig_time, plan)
        query2runtime[query] = query_orig_time
        orig_total_time += query_orig_time
        text_oneline = query.text.replace('\n', ' ')
        logging.info(f"{i}/{len(workload.queries)} {(query, query_orig_time)} text: {text_oneline}")
    logging.info(f"workload orig_total_time: {orig_total_time}")
    return orig_total_time, query2runtime, sql2plan
    
