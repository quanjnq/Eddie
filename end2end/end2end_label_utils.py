import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from selection.dbms.postgres_dbms import PostgresDatabaseConnector
from selection.algorithms.auto_admin_algorithm import AutoAdminAlgorithm
from selection.algorithms.anytime_algorithm import AnytimeAlgorithm

from selection.workload import Workload, Query
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
        actual_runtimes, plan = db_connector.exec_query(query, timeout=timeout, print_err=False)
        if actual_runtimes is None and k > 0:
            return None, None
        each_times.append(actual_runtimes)
        
    # each_times.remove(max(each_times))
    if times > 1:
        avg_times = sum(each_times[1:]) / len(each_times[1:])
    else:
        avg_times = each_times[0]
    return avg_times, plan
    
    
def clear_all_real_indexes(db_connector):
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
        
def get_existed_inds(db_connector):
    stmt = "select indexname from pg_indexes where schemaname='public'"
    indexes = db_connector.exec_fetch(stmt, one=False)
    for index in indexes:
        pass
    return []

def is_relavent_q_ind(q, ind):
    cols = ind.split("#")[1].split(",")
    # q_cols = set([q_col.name for q_col in q.columns])
    i_cols = set([i_col for i_col in cols])
    for c in i_cols:
        if c in q.text:
            return True
    return False

def get_real_label(query, index_config, db_connector, query_plan):
    
    ind_cfg = [ind for ind in index_config if is_relavent_q_ind(query, ind)]
    # orig_time_cost = query_plan[0]["Plan"]["Actual Total Time"]
    orig_time_cost = query_plan["Actual Total Time"]
    
    create_indexes(db_connector, ind_cfg)
    ind_time_cost, plan = get_avg_runtimes(query, orig_time_cost, db_connector)
    if not ind_time_cost:
        ind_time_cost = orig_time_cost
    clear_all_real_indexes(db_connector)
    logging.info(f"ind_time_cost{ind_time_cost}, orig_time_cost{orig_time_cost}, {sorted(ind_cfg)}")
    return (orig_time_cost - ind_time_cost) / orig_time_cost

def get_q_inds2label(data_path):
    q_inds2label = {}
    with open(data_path, "rb") as f:
        samples = pickle.load(f)
    for sample_dict in samples:
        
        if type(sample_dict) is dict:
            query = sample_dict["query"]
            init_index_comb = sample_dict["init_index_config"]
            index_comb = sample_dict["index_config"]
            orig_plan = sample_dict["orig_plan"]
            with_index_plan = sample_dict["with_index_plan"]
            orig_times = sample_dict["orig_runtimes"]
            with_index_runtimes = sample_dict["with_index_runtimes"]
            group_num = sample_dict["group_num"]
        else:
            query = sample_dict[0]
            init_index_comb, index_comb = sample_dict[1]
            orig_times, orig_plan = sample_dict[2]
            with_index_runtimes, with_index_plan = sample_dict[3]
        
        if with_index_runtimes is None:
            with_index_runtimes, with_index_plan = orig_times, orig_plan
        if orig_times > 0.0 and orig_times >= with_index_runtimes:
            label = (orig_times - with_index_runtimes) / orig_times
        else:
            label = 0
        
        sql = query.text
        sql_md5 = string_to_md5(sql)
        
        indexes = []
        for index_str in index_comb:
            indexes.append(index_str)
        indexes.sort()
        q_inds2label[(sql, tuple(indexes))] = label
    return q_inds2label