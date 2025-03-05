import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from selection.dbms.postgres_dbms import PostgresDatabaseConnector
from selection.algorithms.auto_admin_algorithm import AutoAdminAlgorithm
from selection.algorithms.anytime_algorithm import AnytimeAlgorithm
from selection.end2end_utils import *

from selection.workload import Workload
from selection import cost_evaluation_whatif
import logging
import pickle
import itertools
import random
import torch

import time
from data_process.data_process import *
from util import random_util
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')




def calculate_indexes(workload, algorithm_type, db_connector, max_indexes=3):
    logging.info(f"use algorithm_type: {algorithm_type}")
    st = time.time()
    
    cost_eval = cost_evaluation_whatif.CostEvaluation(db_connector)
    
    if algorithm_type == DTA:
        dta_parameters = {'max_index_width': 2, 'budget_MB': 500, 'max_runtime_minutes': 10, "CostEvaluation": cost_eval}
        algorithm = AnytimeAlgorithm(db_connector, dta_parameters)  # dta
    elif algorithm_type == AUTO_ADMIN:
        auto_admin_parameters = { "max_indexes": max_indexes, "max_indexes_naive": 1, "max_index_width": 2, "CostEvaluation": cost_eval}
        algorithm = AutoAdminAlgorithm(db_connector, auto_admin_parameters)
    
    indexes = algorithm.calculate_best_indexes(workload)

    ind_cfg = []
    for ind in indexes:
        ind_str = ind.table().name + "#" + ",".join([c.name for c in ind.columns])
        ind_cfg.append(ind_str)
    ind_cfg.sort()
    logging.info(f"calculate_indexes ratio: {algorithm.cost_evaluation.cache_hits} / {algorithm.cost_evaluation.cost_requests} = {algorithm.cost_evaluation.cache_hits / algorithm.cost_evaluation.cost_requests}")
    logging.info(f"whatif calculate_indexes: {ind_cfg}")
    logging.info(f"calculate_indexes runtime: {time.time() - st} s")
    return ind_cfg


def run_main(dataset_path, algorithm_type= AUTO_ADMIN, db_name = "indexselection_tpcds___10"):
    
    db_connector = PostgresDatabaseConnector(db_name, autocommit=True)
    db_connector.drop_indexes()
    db_connector.commit()
    
    test_workload = get_test_workload(dataset_path, k_flod=10)
    orig_total_time, query2runtime = get_workload_runtime(test_workload, db_connector)
    index_config = calculate_indexes(test_workload, algorithm_type, db_connector)
    total_runtime_with_indexes, regression_query = evaluate_indexes_quality(test_workload, index_config, query2runtime, db_connector)
    logging.info("WHAT-IF")
    logging.info(f"{algorithm_type} (ms) orig_total_time time: {orig_total_time}")
    logging.info(f"{algorithm_type} (ms) total_runtime_with_indexes: {total_runtime_with_indexes}")
    logging.info(f"{algorithm_type} cost saving: (ms) {orig_total_time - total_runtime_with_indexes}")
    logging.info(f"{algorithm_type} cost saving rate: {(orig_total_time - total_runtime_with_indexes) / orig_total_time}")
    logging.info(f"regression_query: {regression_query}")
    logging.info(f"{algorithm_type} total_runtime_with_indexes: {total_runtime_with_indexes/1000} s  orig_total_time time: {orig_total_time/1000} s")
    db_connector.close()
    return total_runtime_with_indexes, regression_query

def eval_what_if(test_workload, algorithm_type, db_connector, query2runtime, orig_total_time, max_indexes=3, times=5):
    logging.info("Strat what_if indexes_quality")
    index_config = calculate_indexes(test_workload, algorithm_type, db_connector, max_indexes=max_indexes)
    total_runtime_with_indexes, regression_query = evaluate_indexes_quality(test_workload, index_config, query2runtime, db_connector, times=times)
    logging.info("WHAT-IF")
    logging.info(f"{algorithm_type} (ms) orig_total_time time: {orig_total_time}")
    logging.info(f"{algorithm_type} (ms) total_runtime_with_indexes: {total_runtime_with_indexes}")
    logging.info(f"{algorithm_type} cost saving: (ms) {orig_total_time - total_runtime_with_indexes}")
    logging.info(f"{algorithm_type} cost saving rate: {(orig_total_time - total_runtime_with_indexes) / orig_total_time}")
    logging.info(f"regression_query: {regression_query}")
    logging.info(f"{algorithm_type} total_runtime_with_indexes: {total_runtime_with_indexes/1000} s  orig_total_time time: {orig_total_time/1000} s")
    logging.info("End what_if indexes_quality")
    return {"runtime_with_indexes(ms)": total_runtime_with_indexes, "regression_querys":regression_query, "regression": len(regression_query), "cost_saving(ms)": orig_total_time - total_runtime_with_indexes, "improve": (orig_total_time - total_runtime_with_indexes) / orig_total_time, "orig_runtime(ms)":orig_total_time }
