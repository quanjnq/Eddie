import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from selection.dbms.postgres_dbms import PostgresDatabaseConnector
from selection.algorithms.auto_admin_algorithm import AutoAdminAlgorithm
from selection.workload import Workload
import logging
import pickle
import itertools
import random
from util.const_util import AUTO_ADMIN_INDCFG, PERTURBED_INDCFG, data_gen_work_dir

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')


def run_autoadmin(workload, K=5, max_index_width=2, db_connector=None):
    auto_admin_parameters = {
    "max_indexes": K,
    "max_indexes_naive": 1,
    "max_index_width": max_index_width
    }
    
    q2bestindcfg = {}
    q2indcfgs = {}
    q2extendindcfgs = {}
    final_sample_list = []
    
    
    for qi, query in enumerate(workload.queries):
        if qi % 50 == 0:
            logging.info(f"Generating indexes for query {qi}")
        w = Workload([query])
        algorithm = AutoAdminAlgorithm(db_connector, auto_admin_parameters)
        autoadmin_indexes = algorithm.calculate_best_indexes(w)
        best_indcfg = []
        for ind in autoadmin_indexes:
            index_str = f"{ind.table()}#{ind.joined_column_names()}"
            best_indcfg.append(index_str)
        best_indcfg = tuple(sorted(best_indcfg))
        
        sub_index_confs = set()
        
        # Enumerates all subsets
        for k in range(1, len(best_indcfg)):
            sub_index_confs |= set(itertools.combinations(best_indcfg, k))
            
        sub_index_confs = list(sub_index_confs)
        sorted_index_confs = []
        for sub_indcfg in sub_index_confs:
            sorted_index_confs.append(tuple(sorted(sub_indcfg)))
        sorted_index_confs.append(best_indcfg)
        
        q2bestindcfg[query] = best_indcfg # Optimal index configuration
        q2indcfgs[query] = sorted_index_confs
        
        # Perturbation index
        perturbed_index_confs_set = set()
        for indexes in sorted_index_confs:
            is_perturbed = False
            perturbed_indexes = []
            for ind in indexes:
                tbl, cols = ind.split("#")
                col_list = cols.split(",")
                if len(col_list) > 1:
                    index_str_extend = f"{tbl}#{col_list[1]},{col_list[0]}"
                    is_perturbed  =True
                    perturbed_indexes.append(index_str_extend)
                else:
                    perturbed_indexes.append(ind)
            if is_perturbed:
                perturbed_index_confs_set.add(tuple(perturbed_indexes))
        
        perturbed_index_confs_list = list(perturbed_index_confs_set)
        sorted_perturbed_index_confs = []
        for indcfg in perturbed_index_confs_list:
            sorted_perturbed_index_confs.append(tuple(sorted(indcfg)))
        
        q2extendindcfgs[query] = sorted_perturbed_index_confs
    
    for query in q2bestindcfg:
        final_sample_list.append({"query":query, AUTO_ADMIN_INDCFG: q2indcfgs[query], PERTURBED_INDCFG: q2extendindcfgs[query], "best_index_config":q2bestindcfg[query]})
    
    return final_sample_list


def main(run_params):
    seed = 0
    random.seed(seed)
    logging.info(f"Run run_gen_query_indexes seed {seed} param: {run_params}")
    db_name = run_params["db_name"]
    workload_path = run_params["workload_path"]
    if "conn_cfg" in run_params:
        conn_cfg = run_params["conn_cfg"]
        db_connector = PostgresDatabaseConnector(db_name, host=conn_cfg["host"],  port=conn_cfg["port"],  user=conn_cfg["user"], password=conn_cfg["password"])
    else:
        db_connector = PostgresDatabaseConnector(db_name)
    db_connector.drop_indexes()
    db_connector.commit()

    with open(workload_path, 'rb') as f:
        workload = pickle.load(f)
    
    sample_list = run_autoadmin(workload, K=5, max_index_width=2, db_connector=db_connector)
    logging.info(f"Example {sample_list[0]}")
    
    len_autoadmin = sum([len(sp[AUTO_ADMIN_INDCFG]) for sp in sample_list])
    len_autoadmin_extend = sum([len(sp[PERTURBED_INDCFG]) for sp in sample_list])
    
    workload_def = workload_path.split("/")[-1].split(".")[0]
    save_dir_path = data_gen_work_dir + "candidate_indexes/"
    if not os.path.exists(save_dir_path):
        os.makedirs(save_dir_path)
    query_indexes_path = save_dir_path + f"{workload_def}__{db_name}_atadm{len_autoadmin}_ext{len_autoadmin_extend}.pickle"
    with open(query_indexes_path, "wb") as f:
        pickle.dump(sample_list, f)
    
    try:
        db_connector.close()
    except Exception as e:
        logging.info(f"db_connector close excetion {e}")
    logging.info(f"Save candidate indexes to: {query_indexes_path}")
    
    return query_indexes_path


if __name__ == "__main__":
    run_params = {"db_name": "indexselection_tpcds___10", "workload_path": "./sql_gen/data/workload/indexselection_tpcds___10_10q.pickle"}
    main(run_params)
    
