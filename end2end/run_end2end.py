import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from selection.dbms.postgres_dbms import PostgresDatabaseConnector
from selection.algorithms.auto_admin_algorithm import AutoAdminAlgorithm

from end2end.end2end_utils import *
from selection.workload import Workload
import logging
import random
import torch
from eddie.eddie_model import Eddie
from run_eddie import ModelArgs as Args_eddie
from util import random_util
from end2end import cost_evaluation_eddie, cost_evaluation_lib, cost_evaluation_whatif

from run_lib import ModelArgs as Args_lib
from lib_est.lib_model import make_model, self_attn_model
import argparse


def get_cost_evaluation(model_name, exp_id, version, fold_i, db_connector, db_stats, sql2plan, checkpoints_path):
    model_path = checkpoints_path.format(model_name=model_name)
    model_path = os.path.join(model_path, f"fold_{fold_i}.pth")
    
    logging.info(f"load model from {model_path}")
    if model_name == "eddie":
        checkpoint = torch.load(model_path, map_location="cpu")
        args = Args_eddie()
        eddie_model = Eddie(max_sort_col_num=args.max_sort_col_num, max_output_col_num=args.max_output_col_num, max_predicate_num=args.max_predicate_num, \
                                disable_idx_attn=args.disable_idx_attn)
        eddie_model.load_state_dict(checkpoint["model"])
        cost_evaluation = cost_evaluation_eddie.CostEvaluation(db_connector = db_connector, db_stat=db_stats, model=eddie_model, sql2plan=sql2plan)
    elif model_name == "lib":
        args = Args_lib()
        encoder_model, pooling_model = make_model(args.dim1, args.n_encoder_layers, args.dim3, args.n_heads, dropout=args.dropout_r)
        lib_model = self_attn_model(encoder_model, pooling_model, 12, args.dim1, args.dim2)
        checkpoint = torch.load(model_path, map_location="cpu")
        lib_model.load_state_dict(checkpoint["model"])
        cost_evaluation = cost_evaluation_lib.CostEvaluation(db_connector = db_connector, db_stat=db_stats, model=lib_model)
    elif model_name == "postgresql":
        cost_evaluation = cost_evaluation_whatif.CostEvaluation(db_connector)
    else:
        logging.error(f"Unknow model_name {model_name}")
        cost_evaluation = None
    return cost_evaluation
    
    
def eval_end2end(model_name, eval_workload, query2runtime, orig_total_time, db_connector, cost_evaluation):
    logging.info(f"Strat eval {model_name}")
    auto_admin_parameters = { "max_indexes": 3, "max_indexes_naive": 1, "max_index_width": 2, "CostEvaluation": cost_evaluation}
    algorithm = AutoAdminAlgorithm(db_connector, auto_admin_parameters)
    indexes = algorithm.calculate_best_indexes(eval_workload)
    ind_cfg = []
    for ind in indexes:
        ind_str = ind.table().name + "#" + ",".join([c.name for c in ind.columns])
        ind_cfg.append(ind_str)
    ind_cfg.sort()
    logging.info(f"calculated indexes by {model_name}: {ind_cfg}")
    total_runtime_with_indexes, regression_query = evaluate_indexes_quality(eval_workload, ind_cfg, query2runtime, db_connector)
    result = {"runtime_with_indexes(ms)": total_runtime_with_indexes, "regression_querys":regression_query, "regression_num": len(regression_query), "cost_saving(ms)": orig_total_time - total_runtime_with_indexes, "improve": (orig_total_time - total_runtime_with_indexes) / orig_total_time, "orig_runtime(ms)":orig_total_time }
    logging.info(f"Eval {model_name} score: {result}")
    logging.info(f"End eval {model_name}")
    return result


def calc_score(result_list):
    total_runtime_with_indexes_time = 0
    total_workload_regression = 0
    total_cost_saveing = 0
    total_orig_runtime = 0
    avg_improve = 0
    regression_query_cnt = 0
    improve_list = []
    for res in result_list:
        runtime_with_indexes = res["runtime_with_indexes(ms)"]
        orig_runtime = res["orig_runtime(ms)"]
        total_runtime_with_indexes_time += runtime_with_indexes
        total_orig_runtime += orig_runtime
        total_cost_saveing += res["cost_saving(ms)"]
        total_workload_regression += 1 if orig_runtime < runtime_with_indexes else 0
        improve_list.append(res["improve"])
        regression_query_cnt += res["regression_num"]
    improve0_5 = 0
    improve5_20 = 0
    improve20_50 = 0
    improve50 = 0
    regress = 0
    for impove in improve_list:
        if 0 <= impove < 0.05:
            improve0_5 += 1
        elif 0.05 <= impove < 0.20:
            improve5_20 += 1
        elif 0.20 < impove < 0.50:
            improve20_50 += 1
        elif 0.50 <= impove:
            improve50 += 1
        else:
            regress += 1
        
    distribition = {"regression": regress, '0%-5%': improve0_5, '5%-20%': improve5_20, '20%-50%': improve20_50, '>50%': improve50, 'total': len(improve_list)}
    avg_improve = sum(improve_list) / len(improve_list)
    score = {"total_cost_saving":total_cost_saveing, "avg_improve": avg_improve, "distribition": distribition, "total_runtime_with_indexes_time(ms)": total_runtime_with_indexes_time, "total_orig_runtime(ms)": total_orig_runtime, "regression": total_workload_regression, "regression_query_cnt": regression_query_cnt, "improve_list":improve_list}
    return score
    

from util.log_util import setup_logging
def main(run_config):
    random_util.seed_everything(0)
    
    model_names = run_config["run_models"]
    dataset_path = run_config["dataset_path"]
    version = run_config.get("version", "v1")
    exp_id = run_config.get("exp_id", dataset_path.split('.')[0])
    checkpoints_path = run_config.get("checkpoints_path", "./checkpoints/tpcds__end2end__{model_name}_v1")

    db_name = run_config["db_name"]
    db_satats_path = run_config["db_stat_path"]
    
    
    log_id = exp_id + "_" + version
    setup_logging(log_id)
    
    db_connector = PostgresDatabaseConnector(db_name, autocommit=True)
    db_connector.drop_indexes()
    db_connector.commit()
    
    
    with open(db_satats_path, "r") as f:
        db_stats = json.load(f)
    
    workload_size = 5
    max_indexes = 3
    k_fold = 5
    
    model2result_list = {}
    
    for fold_i in range(k_fold):
        logging.info(f"========== fold {fold_i}/{k_fold} =========")
        logging.info(f"loading dataset from {dataset_path}")
        test_workload = get_test_workload(dataset_path, k_flod=k_fold, selected_fold=fold_i, db_satats_path=db_satats_path)
        logging.info(f"flod {fold_i} w_size: {len(test_workload.queries)}")
        
        queries = test_workload.queries[:]
        queries.sort(key=lambda q: q.text)
        random.shuffle(queries)
        
        qs_list = split_into_groups(queries, workload_size)
        
        for i, qs in enumerate(qs_list):
            logging.info(f"========== workload group {i}/{len(qs_list)} - fold {fold_i}/{k_fold} =========")
            
            group_workload = Workload(qs)
            logging.info(f"use workload_size: {len(group_workload.queries)} max_indexes: {max_indexes}")
            logging.info(f"start exec workload without indexes")
            db_connector.drop_indexes()
            orig_total_time, query2runtime, sql2plan = get_workload_runtime(group_workload, db_connector)
            
            for model_name in model_names:
                if model_name not in model2result_list:
                    model2result_list[model_name] = []
                cost_evaluation = get_cost_evaluation(model_name, exp_id, version, fold_i, db_connector, db_stats, sql2plan, checkpoints_path)
                result = eval_end2end(model_name, group_workload, query2runtime, orig_total_time, db_connector, cost_evaluation)
                model2result_list[model_name].append(result)
            
    db_connector.drop_indexes()
    db_connector.close()
    
    # print eval result score
    for model_name in model2result_list:
        score = calc_score(model2result_list[model_name])
        logging.info(f"{model_name}: {score}")

    
if __name__ == "__main__":
   
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, default='./datasets/tpcds__end2end.pickle')
    parser.add_argument('--run_models', type=str, default='eddie')
    parser.add_argument('--db_name', type=str, default='indexselection_tpcds___10')
    parser.add_argument('--db_stat_path', type=str, default='./db_stats_data/indexselection_tpcds___10_stats.json')
    parser.add_argument('--checkpoints_path', type=str, default='./checkpoints/tpcds__end2end__{model_name}_v1')
    
    args = parser.parse_args()
    run_cfg = vars(args)
    run_cfg['run_models'] = run_cfg['run_models'].split(',')

    main(run_cfg)

