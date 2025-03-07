
import pickle
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pickle


import logging
import sys
from selection.dbms.postgres_dbms import PostgresDatabaseConnector
from util.workload_util import tran_workload
import json
import random
import re


def gen_new_predicate(tbl_col, db_stat, use_alise):
    cmp_ops = ['<', '>', '=', '>=', '<=', '<>']
    if tbl_col in db_stat:
        # get value
        col_stat = db_stat[tbl_col]
        most_val_list = col_stat["most_common_vals"] if len(col_stat["most_common_vals"]) > 0 else [0]
        comp_num = random.choice(col_stat["histogram_bounds"] if len(col_stat["histogram_bounds"]) > 0 else most_val_list) 
        if use_alise:
            tbl_ali = ""
            for tok in tbl_col.split(".")[0].split("_"):
                tbl_ali = tbl_ali + tok[0]
        tbl_col_name = f"{tbl_ali}.{tbl_col.split('.')[1]}" if use_alise else tbl_col
        new_predicate = f"{tbl_col_name} {random.choice(cmp_ops)} {comp_num}"
        return new_predicate
    return None

    
def gen_predicate(tbl_name, tbl_col=None, db_stat=None, use_alise=False):
    if tbl_col:
        if tbl_col.split(".")[0] == tbl_name:
            if db_stat[tbl_col]["data_type"] in ["integer", "real", "bigint", "numeric", "decimal", "smallint", "double precision"]:
                return gen_new_predicate(tbl_col, db_stat, use_alise)
    colnames = []
    for tbl_col in db_stat:
        if tbl_col.split(".")[0] == tbl_name:
            if db_stat[tbl_col]["data_type"] in ["integer", "real", "bigint", "numeric", "decimal", "smallint", "double precision"]:
                colnames.append(tbl_col)
    seled_tbl_col = random.choice(colnames)
    return gen_new_predicate(seled_tbl_col, db_stat, use_alise)
    

def vary_query(query_text, tbl_names, db_connector, db_stat, use_alise):
    where_token_pattern = r"\b[wW][hH][eE][rR][eE]\b"
    str_li = re.split(where_token_pattern, query_text)
    # to the location of each where
    for i in range(len(str_li)-1):
        # to each possible table
        for tbl_name in tbl_names:
            if tbl_name not in query_text:
                continue
            # Take an predicate
            predicate = gen_predicate(tbl_name, db_stat=db_stat, use_alise=use_alise)
            new_where_predicate = f" where {predicate} and "
            left_str = "where".join(str_li[:i+1])
            right_str = "where".join(str_li[i+1:])
            new_query_text = left_str + new_where_predicate + right_str
            # Check the semantics
            statement = f"explain {new_query_text}"
            try:
                db_connector.exec_fetch(statement, one=True)
                return new_query_text
            except:
                pass


def gen_vary_query_workload(w_path, stat_path, database_name, workload_name, conn_cfg=None):
    random.seed(0)
    use_alise = True if workload_name == "imdb" else False
    with open(stat_path, "r") as f:
        db_stat = json.load(f)
        
    tbl_names = sorted(list(set([tbl_col.split(".")[0] for tbl_col in db_stat])))
    if conn_cfg:
        db_connector = PostgresDatabaseConnector(database_name, host=conn_cfg["host"],  port=conn_cfg["port"],  user=conn_cfg["user"], password=conn_cfg["password"], autocommit=True)
    else:
        db_connector = PostgresDatabaseConnector(database_name, autocommit=True)

    with open(w_path, "rb") as f:
        w = pickle.load(f)
    qs = w.queries

    query_text_li = []
    for q in qs:
        vary_text = vary_query(q.text, tbl_names, db_connector, db_stat, use_alise)
        if vary_text:
            query_text_li.append(vary_text)
        else:
            query_text_li.append(q.text)
    new_workload = tran_workload(query_text_li, db_connector)
    fn = w_path.split("/")[-1].split(".")[0]
    sfp = f"./sql_gen/data/workload/{fn}_new_predicate.pickle"
    dst_dir = os.path.dirname(sfp)
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
    with open(sfp, "wb") as f:
        pickle.dump(new_workload, f)
        logging.info(f"Saved vary query workload: {sfp}")
        
    db_connector.close()
    return sfp

