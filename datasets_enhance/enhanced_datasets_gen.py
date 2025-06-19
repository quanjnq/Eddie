import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import argparse
from pathlib import Path
import os
from selection.dbms.postgres_dbms import PostgresDatabaseConnector
from selection.algorithms.auto_admin_algorithm import AutoAdminAlgorithm

from selection.workload import Workload
import logging
import pickle
import itertools
import time
from data_process.data_process import *
from util import random_util
from selection import cost_evaluation_enhance

log_format = '%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, force=True)
random_util.seed_everything(0)

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


def gen_enhanced_datasets(conn_cfg):
    dataset_path = run_cfg["enhanced_dataset_path"]
    with open(dataset_path, "rb") as f:
        src_data = pickle.load(f)
    src_qs = [sample[0] for sample in src_data]
    queries = list(set(src_qs))
    k = 5
    
    db_connector = PostgresDatabaseConnector(conn_cfg["database_name"], host=conn_cfg["host"],  port=conn_cfg["port"],  user=conn_cfg["user"], password=conn_cfg["password"], autocommit=True)
    sample_dir = "./samples_enhanced"
    Path(sample_dir).mkdir(parents=True, exist_ok=True)
    db_connector.drop_indexes()
    db_connector.commit()
    final_all_group_samples = []
    
    for i in range(30):
        logging.info(f"start group: {i}")
        sp_qs = random.sample(queries, k)
        workload = Workload(sp_qs)
        try:
            a, b, sql2plan = get_workload_runtime(workload, db_connector=db_connector)
        except Exception:
            continue
        
        
        cost_eval = cost_evaluation_enhance.CostEvaluation(db_connector, cost_estimation="actual_runtimes", sql2plan=sql2plan, workload=workload, group_num=i, w_path=dataset_path)
        auto_admin_parameters = { "max_indexes": 3, "max_indexes_naive": 1, "max_index_width": 2, "CostEvaluation": cost_eval}
        algorithm = AutoAdminAlgorithm(db_connector, auto_admin_parameters)
        indexes = algorithm.calculate_best_indexes(workload)
        fp = f"{sample_dir}/group_{i}_{len(cost_eval.all_samples)}.pickle"
        with open(fp, "wb") as f:
            pickle.dump(cost_eval.all_samples, f)
            print(fp)
            final_all_group_samples.extend(cost_eval.all_samples)
        print(indexes)
    ffp = f"{sample_dir}/group_{len(final_all_group_samples)}.pickle"
    with open(ffp, "wb") as f:
        pickle.dump(cost_eval.all_samples, f)
        print(fp)
        


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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run dataset enhancements')
    
    # Required arguments
    parser.add_argument('--enhanced_dataset_path', type=str, required=True, help='The path of the dataset that will be enhanced.')
    
    # Optional arguments
    parser.add_argument('--database_name', type=str, help='The database name used for the connection')
    parser.add_argument('--host', type=str, help='The host address used for the connection')
    parser.add_argument('--port', type=str, help='The port used for the connection')
    parser.add_argument('--user', type=str, help='The username used for the connection')
    parser.add_argument('--password', type=str, help='The password used for the connection')
    
    args = parser.parse_args()
    run_cfg = vars(args)  # Convert args directly to dictionary
    gen_enhanced_datasets(run_cfg)
    

''' example
python datasets_enhance/enhanced_datasets_gen.py \
    --host database_name \
    --host localhost \
    --port 54321 \
    --user postgres \
    --password your_password
    --enhanced_dataset_path ./datasets/tpcds__base_w_init_idx.pickle
'''
