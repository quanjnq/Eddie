import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from datasets_gen import gen_candidate_indexes as gq_inds
from datasets_gen import gen_samples as gspl
import pickle
import logging
import json
import shutil
from datasets_gen.gen_init_idx import gen_with_init_idx_samples
from datasets_gen.vary_query import gen_vary_query_workload
from util.const_util import datasets_savr_dir, AUTO_ADMIN_INDCFG, PERTURBED_INDCFG, RANDOMWALK_INDCFG, dataset_gen_config_path
from datasets_gen.randomwalk_enhance_sample import enhance_index_for_dataset
import argparse
from util.log_util import setup_logging

def get_dataset_path(workload_name, exp_id):
    return datasets_savr_dir + f"{workload_name}__{exp_id}.pickle"

def gen_sample(db_name, w_path, indextype, conn_cfg=None):
    # Only for AUTO_ADMIN_INDCFG and PERTURBED_INDCFG
    params = {"db_name": db_name, "workload_path": w_path,}
    if conn_cfg:
        params["conn_cfg"] = conn_cfg
    query_indexes_path = gq_inds.main(params)
    
    gen_samples_params = {"db_name": db_name, "query_indcfgs_path": query_indexes_path, "indextype_and_breakpoint": [(indextype, 0)]}
    if conn_cfg:
        gen_samples_params["conn_cfg"] = conn_cfg
    sample_path_dict = gspl.main(gen_samples_params)
    autoadmin_sample_path = sample_path_dict[indextype]
    return autoadmin_sample_path


def gen_base_wo_init_idx(cfg):
    logging.info(f"Start base_wo_init_idx use cfg: {cfg}")
    workload_name = cfg["workload_name"]
    exp_id = cfg["exp_id"]
    db_name = cfg["db_name"]
    w_path = cfg["workload_path"]
    autoadmin_sample_path = gen_sample(db_name, w_path, AUTO_ADMIN_INDCFG, cfg["conn_cfg"])
    dataset_path = get_dataset_path(workload_name, exp_id)
    dst_dir = os.path.dirname(dataset_path)
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
    shutil.copy(autoadmin_sample_path, dataset_path)
    logging.info(f"Saved dataset: {dataset_path}")
    return dataset_path
    

def gen_base_w_init_idx(cfg):
    logging.info(f"Start base_w_init_idx use cfg: {cfg}")
    workload_name = cfg["workload_name"]
    exp_id = cfg["exp_id"]
    db_name = cfg["db_name"]
    base_exp_id = cfg["base_exp_id"]
    w_path = cfg["workload_path"]
    base_dataset_path = get_dataset_path(workload_name, base_exp_id)
    dataset_path = get_dataset_path(workload_name, exp_id)
    if not os.path.exists(base_dataset_path):
        base_dataset_path = gen_base_wo_init_idx(workload_name, base_exp_id, db_name, w_path)
    gen_with_init_idx_samples(base_dataset_path, dataset_path)
    return dataset_path


def gen_perturb_idx(cfg):
    logging.info(f"Start perturb_idx use cfg: {cfg}")
    workload_name = cfg["workload_name"]
    exp_id = cfg["exp_id"]
    db_name = cfg["db_name"]
    base_exp_id = cfg["base_exp_id"]
    w_path = cfg["workload_path"]
    
    base_dataset_path = get_dataset_path(workload_name, base_exp_id)
    dataset_path = get_dataset_path(workload_name, exp_id)
    if not os.path.exists(base_dataset_path):
        base_dataset_path = gen_base_wo_init_idx(workload_name, base_exp_id, db_name, w_path)
    perturb_sample_path = gen_sample(db_name, w_path, PERTURBED_INDCFG , cfg["conn_cfg"])
    with open(base_dataset_path, "rb") as f:
        base_smpls = pickle.load(f)
        
    with open(perturb_sample_path, "rb") as f:
        perturb_smpls = pickle.load(f)
    
    all_samples = base_smpls + perturb_smpls
    
    dst_dir = os.path.dirname(dataset_path)
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
    with open(dataset_path, "wb") as f:
        pickle.dump(all_samples, f)
        logging.info(f"Saved dataset: {dataset_path}")
    return dataset_path


def gen_vary_query(cfg):
    logging.info(f"Start vary_query use cfg: {cfg}")
    workload_name = cfg["workload_name"]
    exp_id = cfg["exp_id"]
    dataset_path = get_dataset_path(workload_name, exp_id)
    db_name = cfg["db_name"]
    w_path = cfg["workload_path"]
    db_stat_path = cfg["db_stat_path"]
    
    vary_query_workload_path = gen_vary_query_workload(w_path, db_stat_path, db_name, workload_name)
    autoadmin_sample_path = gen_sample(db_name, vary_query_workload_path, AUTO_ADMIN_INDCFG, cfg["conn_cfg"])
    gen_with_init_idx_samples(autoadmin_sample_path, dataset_path)
    return dataset_path
    

def gen_vary_stat(cfg):
    logging.info(f"Start vary_stat use cfg: {cfg}")
    workload_name = cfg["workload_name"]
    exp_id = cfg["exp_id"]
    dataset_path = get_dataset_path(workload_name, exp_id)
    db_name = cfg["transfer_db_name"]
    w_path = cfg["workload_path"]
    autoadmin_sample_path = gen_sample(db_name, w_path, AUTO_ADMIN_INDCFG,)
    gen_with_init_idx_samples(autoadmin_sample_path, dataset_path)
    return dataset_path


def gen_end2end(cfg):
    logging.info(f"Start end2end use cfg: {cfg}")
    
    workload_name = cfg["workload_name"]
    exp_id = cfg["exp_id"]
    db_name = cfg["db_name"]
    base_exp_id = cfg["base_exp_id"]
    base_dataset_path = get_dataset_path(workload_name, base_exp_id)
    query_indexes_path = enhance_index_for_dataset(base_dataset_path)
    indextype = RANDOMWALK_INDCFG
    gen_samples_params = {"db_name": db_name, "query_indcfgs_path": query_indexes_path, "indextype_and_breakpoint": [(indextype, 0)]}
    sample_path_dict = gspl.main(gen_samples_params)
    enhanced_sample_path = sample_path_dict[indextype]
    
    with open(base_dataset_path, "rb") as f:
        base_smpls = pickle.load(f)
        
    with open(enhanced_sample_path, "rb") as f:
        enhanced_smpls = pickle.load(f)
    
    all_samples = base_smpls + enhanced_smpls
    
    dataset_path = get_dataset_path(workload_name, exp_id)
    dst_dir = os.path.dirname(dataset_path)
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
    with open(dataset_path, "wb") as f:
        pickle.dump(all_samples, f)
        logging.info(f"Saved dataset: {dataset_path}")
    return dataset_path


def gen_pretrain(cfg):
    exp_id = cfg["exp_id"]
    for wcfg in cfg["workload_cfgs"]:
        path = wcfg["path"]
        workload_name = wcfg["workload_name"]
        db_name = wcfg["db_name"]
        autoadmin_sample_path = gen_sample(db_name, path, AUTO_ADMIN_INDCFG)
        dataset_path = datasets_savr_dir + f"{exp_id}_{workload_name}.pickle"
        dst_dir = os.path.dirname(dataset_path)
        if not os.path.exists(dst_dir):
            os.makedirs(dst_dir)
        shutil.copy(autoadmin_sample_path, dataset_path)
        logging.info(f"Saved dataset: {dataset_path}")
        

def main(conn_cfg):
    
    exp_id2func = {"base_wo_init_idx": gen_base_wo_init_idx, "base_w_init_idx": gen_base_w_init_idx, "perturb_idx": gen_perturb_idx, "vary_query": gen_vary_query, "vary_stat": gen_vary_stat, "end2end": gen_end2end}
    run_config_path = dataset_gen_config_path
    
    with open(run_config_path, "r") as f:
        run_config = json.load(f)
    run_config.update(conn_cfg)    
    run_config["conn_cfg"] = conn_cfg
    log_id = run_config["run_id"] if "run_id" in run_config else "data_gen"
    setup_logging(log_id)
    logging.info(f"Load run config from: {run_config_path}")
    db_id2info = {}
    for db_cfg in run_config["databases"]:
        db_id2info[db_cfg["db_id"]] = db_cfg
    
    # For main and end2end
    for workload_cfg in run_config["workloads"]:
        if not workload_cfg["activated"]:
            continue
        exp_id_list = workload_cfg["exp_id_list"]
        db_id = workload_cfg["db_id"]
        cfg = {}
        cfg.update(workload_cfg)
        cfg["conn_cfg"] = conn_cfg
        cfg["db_name"] = db_id2info[db_id]["db_name"]
        cfg["db_stat_path"] = db_id2info[db_id]["db_stat_path"]
        if "transfer_db_id" in workload_cfg:
            cfg["transfer_db_name"] = db_id2info[workload_cfg["transfer_db_id"]]["db_name"]
        for exp_id in exp_id_list:
            cfg["exp_id"] = exp_id
            if exp_id not in exp_id2func:
                logging.info(f"exp_id: {exp_id} not in exp_id2func")
                continue
            exp_id2func[exp_id](cfg)
    
    # For pretrain
    if "pretrain" in run_config:
        gen_pretrain(run_config["pretrain"])
        

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
    cfg = vars(args)  # Convert args directly to dictionary
    main(cfg)


''' example
python datasets_gen/gen_datasets.py \
    --run_id data_gen_tpcds_1 \
    --host localhost \
    --port 54321 \
    --user postgres \
    --password your_password
'''