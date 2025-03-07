import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import random
import pickle
from selection.dbms.postgres_dbms import PostgresDatabaseConnector
from selection.query_generator import QueryGenerator
from selection.table_generator import TableGenerator
from selection.workload import Workload
from util.const_util import *
from workload_gen.imdb_template import paramize_imdb
from workload_gen.gen_synthetic_query import main as synthetic_main
from workload_gen.gen_zero_shot_workload import main as zero_shot_main
import json
import argparse
import logging
from util.log_util import setup_logging


# tpcds_queries = [2,3,5,7,8,9,12,13,15,17,18,19,20,21,24,25,26,27,28,29,31,33,34,36,37,38,40,41,42,43,44,45,46,47,48,49,50,51,52,53,55,56,57,58,59,60,61,62,63,65,66,68,69,70,71,72,73,75,76,77,78,79,80,82,83,84,85,86,87,88,89,90,91,93,96,97,98,99]
# tpch_queries = [3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 16, 19, 22]
# imdb_queries =[1, 2, 3, 4, 5, 7, 9, 10, 11, 12, 13, 14, 15, 17, 18, 19, 20, 23, 25, 26, 27, 29, 32, 35, 38, 39, 40, 42, 43, 46, 48, 49, 50, 51, 52, 53, 54, 58, 59, 60, 62, 63, 64, 65, 66, 67, 68, 69, 70, 72, 73, 74, 75, 76, 77, 78, 79, 81, 83, 84, 85, 86, 87, 88, 89, 91, 93, 95, 96, 97, 98, 99, 100, 102, 103, 105, 107, 111, 112, 113]

# Process validation
tpcds_queries = [2,]
tpch_queries = [3, ]
imdb_queries =[1, ]

cfg_dict = {TPCDS: {"workload_name":TPCDS, "gen_query_times": 5, "scale_factor": 10, "queries": tpcds_queries, },
          TPCH:    {"workload_name":TPCH, "gen_query_times": 80, "scale_factor": 10, "queries": tpch_queries, },
          IMDB:    {"workload_name":IMDB, "gen_query_times": 3, "database_name": "imdbload", "db_stat_path": "/data/dataset_gen/stats_data/imdbload_stats.json", "queries": imdb_queries, "imdb_dir": imdb_dir},
          TPCDS_PLUS: {"workload_name":TPCDS_PLUS,"base_workload_name": TPCDS, "query_cnt": 3000,  "database_name": "indexselection_tpcds___10"},
          TPCH_PLUS: {"workload_name":TPCH_PLUS, "base_workload_name": TPCH, "query_cnt": 3000,  "database_name": "indexselection_tpch___10"},
          IMDB_PLUS: {"workload_name":IMDB_PLUS,"base_workload_name": IMDB, "query_cnt": 3000,  "database_name": "imdbload"},
          }

pretrain_workload_cfgs = [
    {"workload_name": "baseball", "db_name":"baseball_scaled", "query_cnt": 1000},
    {"workload_name": "employee", "db_name":"employee_scaled", "query_cnt": 1000},
    # ...
    ]
    

def save_workload(workload, pickle_path):
    pickle.dump(workload, open(pickle_path, "wb"))
    logging.info(f"Saved workload: {pickle_path}")
    return pickle_path
    
def get_workload_path(workload_name):
    
    w_file_name = f"workload_{workload_name}.pickle"
    pickle_path = workload_save_path + w_file_name

    dst_dir = os.path.dirname(pickle_path)
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
    return pickle_path
    
def gen_tpc(config):
    
    random.seed(0)
    workload_name = config["workload_name"]
    scale_factor = config["scale_factor"]
    
    if "conn_cfg" in config:
        conn_cfg = config["conn_cfg"]
        db_connector = PostgresDatabaseConnector(None, host=conn_cfg["host"],  port=conn_cfg["port"],  user=conn_cfg["user"], password=conn_cfg["password"], autocommit=True)
    else:
        db_connector = PostgresDatabaseConnector(None, autocommit=True)
    
    logging.info(f"gen {workload_name} use config:", config)
    
    table_generator = TableGenerator(
        workload_name, scale_factor, db_connector, None
    )
    db_connector = PostgresDatabaseConnector(table_generator.database_name(), autocommit=True)
    if "queries" not in config:
        config["queries"] = None # All queries

    query_generator = QueryGenerator(
        workload_name,
        scale_factor,
        db_connector,
        config["queries"],
        table_generator.columns,
        config["gen_query_times"]
    )
    workload = Workload(query_generator.queries)
    db_connector.close()
    return save_workload(workload, get_workload_path(workload_name))
    

def gen_imdb(cfg):
    # imdb query list dir
    base_dir = cfg["imdb_dir"]
    queries = set(cfg["queries"])
    database_name = cfg["database_name"]
    db_stat_path = cfg["db_stat_path"]
    gen_query_times = cfg["gen_query_times"]
    workload_name = cfg["workload_name"]
    logging.info(f"gen {workload_name} use config:", cfg)
    db_stat = json.load(open(db_stat_path, "r"))

    files = [os.path.join(base_dir, file) for file in os.listdir(base_dir)]
    fns = [fn for fn in files if fn.endswith("sql") and "schema" not in fn and "fkindexes" not in fn]

    sqls = []
    for fn in fns:
        with open(fn, "r") as f:
            sql = "".join(f.readlines())
            sqls.append(sql)

    selected_sql_list = []
    for qi, sql in enumerate(sqls):
        nr = qi+1
        if queries:
            if nr in queries:
                selected_sql_list.append((nr, sql))
        else:
            selected_sql_list.append((nr, sql))
    conn_cfg = cfg["conn_cfg"] if "conn_cfg" in cfg else None
    workload_imdb = paramize_imdb(selected_sql_list, database_name, db_stat, times=gen_query_times, conn_cfg=conn_cfg)
    return save_workload(workload_imdb, get_workload_path(workload_name))

def gen_workload_plus(cfg):
    workload_name = cfg["workload_name"]
    base_workload_name = cfg["base_workload_name"]
    query_cnt = cfg["query_cnt"]
    dbname = cfg["database_name"]
    
    gen_synthetic_query_params = {"dbname": dbname, "workload_name": base_workload_name, "query_total_count": query_cnt}
    gen_synthetic_query_params["conn_cfg"] = cfg["conn_cfg"]
    synthetic_workload_path = synthetic_main(gen_synthetic_query_params)
    # Merge workload
    templated = pickle.load(open(get_workload_path(base_workload_name), "rb"))
    synthetic = pickle.load(open(synthetic_workload_path, "rb"))
    queries = templated.queries + synthetic.queries
    w = Workload(queries)
    return save_workload(w, get_workload_path(workload_name))


def gen_workload(cfg_dict, func_map, conn_cfg=None):
    for w_name in cfg_dict:
        run_cfg = cfg_dict[w_name]
        run_cfg["conn_cfg"] = conn_cfg
        func_map[w_name](run_cfg)

def main(conn_cfg):
    log_id = conn_cfg["log_id"] if "log_id" in conn_cfg else "workload_gen"
    setup_logging(log_id)
    gen_func_map = {TPCDS: gen_tpc, TPCH: gen_tpc, IMDB: gen_imdb, TPCDS_PLUS: gen_workload_plus, TPCH_PLUS: gen_workload_plus, IMDB_PLUS: gen_workload_plus}
    gen_workload(cfg_dict, gen_func_map, conn_cfg=conn_cfg)
    for cfg in pretrain_workload_cfgs:
        zero_shot_main(cfg["workload_name"], cfg["db_name"], cfg["query_cnt"])
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run Eddie model training and evaluation')
    # Required arguments
    parser.add_argument('--run_id', type=str, required=True, help=' Unique identifier for this run. This will be used for logging naming.')
    
    # Optional arguments
    parser.add_argument('--host', type=str, help='The host address used for the connection')
    parser.add_argument('--port', type=str, help='The port used for the connection')
    parser.add_argument('--user', type=str, help='The username used for the connection')
    parser.add_argument('--password', type=str, help='The password used for the connection')
    
    args = parser.parse_args()
    conn_cfg = vars(args)  # Convert args directly to dictionary
    main(conn_cfg)
    
''' example
python workload_gen/gen_workload.py \
    --run_id workload_gen \
    --host localhost \
    --port 54321 \
    --user postgres \
    --password your_password
'''