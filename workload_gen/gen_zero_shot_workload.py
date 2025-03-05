import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pickle
import json
from selection.dbms.postgres_dbms import PostgresDatabaseConnector
from util.workload_util import tran_workload
from util.const_util import zero_shot_workload_path, workload_name_placehoder, workload_save_path
import random


def get_sqls(workload_name, num, db_connector=None):
    real_workload_path = zero_shot_workload_path.replace(workload_name_placehoder, workload_name)
    with open(real_workload_path, "r") as f:
        wdict = json.load(f)
    
    complex_sqls = [qd['sql'] for qd in wdict["query_list"]]

    if db_connector:
        complex_sqls = [sql for sql in complex_sqls if check_sql(sql, db_connector)]

    sampled_complex_sqls = random.sample(complex_sqls, min(len(complex_sqls), num))

    return list(sampled_complex_sqls)

def check_sql(new_query_text, db_connector):
    statement = f"explain {new_query_text}"
    try:
        db_connector.exec_fetch(statement, one=True)
        return True
    except:
        return False


def tran_and_save_w(workload_name, sqls, db_connector):
    w  = tran_workload(sqls, db_connector)
    w_path = workload_save_path + f"{workload_name}.pickle"
    
    dst_dir = os.path.dirname(w_path)
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
    
    with open(w_path, "wb") as f:
        pickle.dump(w, f)
    print("saved workload pickle:", w_path)
    return w_path



def main(workload_name, database_name, num=1000):
    random.seed(0)
    db_connector = PostgresDatabaseConnector(database_name, autocommit=True)
    sqls = get_sqls(workload_name, num, db_connector=db_connector)
    w_path = tran_and_save_w(workload_name, sqls, db_connector)
    db_connector.close()
    return w_path


if __name__ == "__main__":
    workload_cfgs = [
    {"workload_name": "baseball", "db_name":"baseball_scaled"},
    {"workload_name": "employee", "db_name":"employee_scaled"},
    ]
    for cfg in workload_cfgs:
        main(cfg["workload_name"], cfg["db_name"], 10)
