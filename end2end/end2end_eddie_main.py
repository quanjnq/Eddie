import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from selection.dbms.postgres_dbms import PostgresDatabaseConnector
from selection.algorithms.auto_admin_algorithm import AutoAdminAlgorithm
from selection.algorithms.anytime_algorithm import AnytimeAlgorithm
from selection.end2end_utils import *
from util.metrics_util import calc_qerror
from selection.workload import Workload
import logging
import pickle
import itertools
import random
import torch
from eddie.eddie_model import Eddie
from run_eddie import Args
import time
from util import random_util
from selection import cost_evaluation_eddie
from selection import end2end_whatif, end2end_lib

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')


def calculate_indexes(workload, algorithm_type, db_connector, db_stats, model, sql2plan, data_path=None, enable_invoke_stat=False, use_real_label=False, max_indexes=3):
    logging.info(f"use algorithm_type: {algorithm_type}")
    st = time.time()
    cost_evaluation = cost_evaluation_eddie.CostEvaluation(db_connector = db_connector, db_stat=db_stats, model=model, sql2plan=sql2plan, data_path=data_path, enable_invoke_stat=enable_invoke_stat, use_real_label=use_real_label)
    if algorithm_type == DTA:
        dta_parameters = {'max_index_width': 2, 'budget_MB': 500, 'max_runtime_minutes': 1, "CostEvaluation": cost_evaluation}
        algorithm = AnytimeAlgorithm(db_connector, dta_parameters)  # dta
    elif algorithm_type == AUTO_ADMIN:
        auto_admin_parameters = { "max_indexes": max_indexes, "max_indexes_naive": 1, "max_index_width": 2, "CostEvaluation": cost_evaluation}
        algorithm = AutoAdminAlgorithm(db_connector, auto_admin_parameters)
    
    indexes = algorithm.calculate_best_indexes(workload)

    ind_cfg = []
    for ind in indexes:
        ind_str = ind.table().name + "#" + ",".join([c.name for c in ind.columns])
        ind_cfg.append(ind_str)
    ind_cfg.sort()
    logging.info(f"calculate_indexes ratio: {algorithm.cost_evaluation.cache_hits} / {algorithm.cost_evaluation.cost_requests} = {algorithm.cost_evaluation.cache_hits / algorithm.cost_evaluation.cost_requests}")
    logging.info(f"eddie calculate_indexes: {ind_cfg}")
    logging.info(f"calculate_indexes runtime: {time.time() - st} s")
    
    
    sample_hit_rate = f"{cost_evaluation.hit_cnt}/{cost_evaluation.invoke_cnt}"    
    invoke_benefit_score = calc_qerror(cost_evaluation.real_labels, cost_evaluation.pred_labels)
    logging.info(f"invoke_score: {invoke_benefit_score}")
    logging.info(f"sample_hit_rate: {sample_hit_rate}")
    return ind_cfg


def eval_eddie(test_workload, algorithm_type, db_connector, query2runtime, orig_total_time, data_path, db_stats, sql2plan, enable_invoke_stat=False, selected_fold=0, max_indexes=3, use_real_label=False, times=5):
    logging.info("Strat Eddie indexes_quality")
    fn = data_path.split("/")[-1].split(".")[0]
    model_save_path = f"./checkpoints/{fn}_ours_v13_fold_{selected_fold}.pth"
    logging.info(f"load model from {model_save_path}") 
    args = Args()
    checkpoint = torch.load(model_save_path, map_location="cpu")
    model = Eddie(max_sort_col_num=args.max_sort_col_num)
    model.load_state_dict(checkpoint["model"])
    index_config = calculate_indexes(test_workload, algorithm_type, db_connector, db_stats, model, sql2plan, data_path=data_path, enable_invoke_stat=enable_invoke_stat, max_indexes=max_indexes, use_real_label=use_real_label)
    total_runtime_with_indexes, regression_query = evaluate_indexes_quality(test_workload, index_config, query2runtime, db_connector, times=times)
    logging.info("Eddie")
    logging.info(f"{algorithm_type} (ms) orig_total_time time: {orig_total_time}")
    logging.info(f"{algorithm_type} (ms) total_runtime_with_indexes: {total_runtime_with_indexes}")
    logging.info(f"{algorithm_type} cost saving: (ms) {orig_total_time - total_runtime_with_indexes}")
    logging.info(f"{algorithm_type} cost saving rate: {(orig_total_time - total_runtime_with_indexes) / orig_total_time}")
    logging.info(f"regression_query: {regression_query}")
    logging.info(f"{algorithm_type} total_runtime_with_indexes: {total_runtime_with_indexes/1000} s  orig_total_time time: {orig_total_time/1000} s")
    
    logging.info("End Eddie indexes_quality")
    return {"runtime_with_indexes(ms)": total_runtime_with_indexes, "regression_querys":regression_query, "regression": len(regression_query), "cost_saving(ms)": orig_total_time - total_runtime_with_indexes, "improve": (orig_total_time - total_runtime_with_indexes) / orig_total_time, "orig_runtime(ms)":orig_total_time }
    

def run_main(dataset_path, algorithm_type= AUTO_ADMIN, db_name = "indexselection_tpcds___10", db_stats=None, model=None, wsize=10, max_indexes=3, use_real_label=False, selected_fold=0, test_workload=None, times=5):
    # random_util.seed_everything(0)
    
    db_connector = PostgresDatabaseConnector(db_name, autocommit=True)
    db_connector.drop_indexes()
    db_connector.commit()
    logging.info(f"use w size: {len(test_workload.queries)} max_indexes: {max_indexes}")
    db_connector.drop_indexes()
    orig_total_time, query2runtime, sql2plan = get_workload_runtime(test_workload, db_connector)
    
    # eval
    print()
    res_eddie = eval_eddie(test_workload, algorithm_type, db_connector, query2runtime, orig_total_time, dataset_path, db_stats, sql2plan, enable_invoke_stat=False, selected_fold=selected_fold, max_indexes=max_indexes, use_real_label=use_real_label, times=times)
    print()
    res_whatif = end2end_whatif.eval_what_if(test_workload, algorithm_type, db_connector, query2runtime, orig_total_time, max_indexes=max_indexes, times=times)

    print()
    res_lib = end2end_lib.eval_lib(test_workload, algorithm_type, db_connector, query2runtime, orig_total_time, dataset_path, db_stats, selected_fold=selected_fold, max_indexes=max_indexes, times=times)
    
    db_connector.close()
    return {"eddie": res_eddie, "whatif": res_whatif, "lib": res_lib}

def eval_actual_time(test_workload, index_config, query2runtime, db_connector, orig_total_time):
    total_runtime_with_indexes, regression_query = evaluate_indexes_quality(test_workload, index_config, query2runtime, db_connector)
    logging.info("actual")
    logging.info(f"(ms) orig_total_time time: {orig_total_time}")
    logging.info(f"(ms) total_runtime_with_indexes: {total_runtime_with_indexes}")
    logging.info(f"cost saving: (ms) {orig_total_time - total_runtime_with_indexes}")
    logging.info(f"cost saving rate: {(orig_total_time - total_runtime_with_indexes) / orig_total_time}")
    logging.info("== end actual")
    return {"total_runtime_with_indexes": total_runtime_with_indexes, "regression_query":regression_query}

group2actual_time_indcfg = {
    "1": ['catalog_sales#cs_sold_date_sk', 'inventory#inv_date_sk', 'store_sales#ss_customer_sk'],
'3': ['catalog_sales#cs_order_number,cs_item_sk', 'store_sales#ss_item_sk,ss_store_sk', 'store_sales#ss_ticket_number,ss_net_profit'],
'7': ['catalog_sales#cs_sold_date_sk', 'store_sales#ss_item_sk,ss_ticket_number', 'store_sales#ss_ticket_number,ss_quantity'],
'29': ['inventory#inv_item_sk,inv_quantity_on_hand', 'store_sales#ss_item_sk,ss_ticket_number', 'web_sales#ws_sold_date_sk,ws_promo_sk'],
'20': ['catalog_sales#cs_sold_date_sk', 'customer#c_customer_sk', 'web_sales#ws_sold_date_sk,ws_web_site_sk'],
'19': ['catalog_sales#cs_sold_date_sk', 'store_sales#ss_sold_date_sk', 'store_sales#ss_ticket_number,ss_item_sk'],
'5': ['catalog_sales#cs_sold_date_sk', 'promotion#p_promo_sk', 'store_sales#ss_sold_date_sk,ss_hdemo_sk'],
'0': ['catalog_returns#cr_returned_date_sk', 'catalog_sales#cs_sold_date_sk', 'store_sales#ss_item_sk'],
"12": ['catalog_sales#cs_sold_date_sk,cs_ship_mode_sk', 'date_dim#d_date_sk,d_month_seq', 'date_dim#d_year,d_dom', 'store_sales#ss_sold_date_sk,ss_ext_sales_price', 'web_sales#ws_sold_date_sk,ws_bill_addr_sk'],
}

def main_end(selected_fold=0):
    
    # db_sata_fp = './db_stats_data/indexselection_tpcds___10_stats.json'
    # db_name="indexselection_tpcds___10"
    
    
    db_sata_fp = './db_stats_data/imdbload_stats.json'
    db_name = "imdbload"
    
    # db_sata_fp = './db_stats_data/indexselection_tpch___10_stats.json'
    # db_name = "indexselection_tpch___10"
    
    exp_params = [
        
        # {"data_path": '/home/seven/src/index/index_selection_evaluation/gen_data/workload_tpcds_390_queries_autoadmin_4910_extend_23595_autoadmin_4910_breakpoint0_double_14730.pickle', "algorithm_type": AUTO_ADMIN},
        {"data_path": '/home/seven/src/index/index_selection_evaluation/gen_data/imdb_workload_240_autoadmin_6240_extend_1606_autoadmin6240_sqi0_ics_pairs_12480_randomwalk_7049_sqi0_13289.pickle', "algorithm_type": AUTO_ADMIN},
        # {"data_path": '/home/seven/src/index/index_selection_evaluation/gen_data/workload_tpch_1120_queries_autoadmin_2400_extend_876_autoadmin2400_sqi0_v2_ics_pairs_4800_randomwalk_4800_sqi0_7200.pickle', "algorithm_type": AUTO_ADMIN},
        
        ]
    
    with open(db_sata_fp, "r") as f:
        db_stats = json.load(f)
    
    for param in exp_params:
        # try:
        algorithm_type = param["algorithm_type"]
        data_path = param["data_path"]
        print(f"=========== {algorithm_type}")
        group2res = {}
        res_list = []
        ws = 5
        times = 5
        
        test_workload = get_test_workload(data_path, k_flod=5, selected_fold=selected_fold, rs=-1) # param [rs] todo remove
        logging.info(f"workload size: {ws} from test w_size: {len(test_workload.queries)}")
        queries = test_workload.queries[:]
        queries.sort(key=lambda q: q.text)
        random.shuffle(queries)
        logging.info(f"shuffle queries")
        qs_list = split_into_groups(queries, ws)
        
        for i, qs in enumerate(qs_list):
            # selected_fold = random.randint(0, 4)
            # selected_fold = 1
            logging.info(f"========== epoch {i}/{len(qs_list)} selected_fold {selected_fold}=========")
            res = run_main(test_workload = Workload(qs), algorithm_type=algorithm_type, dataset_path = data_path , db_stats=db_stats,  db_name=db_name, wsize=ws, max_indexes=3, use_real_label=False, selected_fold=selected_fold, times=times)
            res_list.append(res)
            print("eddie_res", res["eddie"])
            print("whatif_res     ", res["whatif"])
            print("lib_res     ", res["lib"])
            logging.info(f"========== End epoch {i} selected_fold {selected_fold}=========")
        # {"eddie": res_eddie, "whatif": res_whatif, "orig": orig_total_time}
        # return {"runtime_with_indexes(ms)": total_runtime_with_indexes, "regression_querys":regression_query, "regression": len(regression_query), "cost_saving(ms)": orig_total_time - total_runtime_with_indexes, "improve": (orig_total_time - total_runtime_with_indexes) / orig_total_time, "orig_runtime(ms)":orig_total_time }
        
        print()
        flod_res = {}
        # for method in ["eddie", "whatif"]:
        for method in ["eddie", "whatif", "lib",]:
        # for method in ["lib",]:
            total_runtime_with_indexes_time = 0
            total_workload_regression = 0
            total_cost_saveing = 0
            total_orig_runtime = 0
            avg_improve = 0
            regression_query_cnt = 0
            improve_list = []
            for res in res_list:
                runtime_with_indexes = res[method]["runtime_with_indexes(ms)"]
                orig_runtime = res[method]["orig_runtime(ms)"]
                total_runtime_with_indexes_time += runtime_with_indexes
                total_orig_runtime += orig_runtime
                total_cost_saveing += res[method]["cost_saving(ms)"]
                total_workload_regression += 1 if orig_runtime < runtime_with_indexes else 0
                improve_list.append(res[method]["improve"])
                regression_query_cnt += res[method]["regression"]
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
            flod_res[method] = score
            logging.info(f"{method}{' '*max(len('eddie')-len(method), 0)}: {score}")
    print(f"#######  end flod {selected_fold}")
    return flod_res
    
if __name__ == "__main__":
    random_util.seed_everything(0)
    flod_res_list = []
    # for flod in range(0,1):
    for flod in range(0,5):
        flod_res = main_end(flod)
        flod_res_list.append(flod_res)
    print()
    print(flod_res_list)
    print()
    # for method in ["eddie", "whatif"]:
    for method in ["eddie", "whatif", "lib",]:
    # for method in ["lib",]:
        total_cost_saving = 0
        imprv_li = []
        for flod_res in flod_res_list:
            total_cost_saving += flod_res[method]["total_cost_saving"]
            imprv_li.extend(flod_res[method]["improve_list"])
        print(method, {"total_cost_saving": total_cost_saving, "avg_improv": sum(imprv_li)/len(imprv_li)})
            
        

# nohup python ./selection/end2end_eddie_main.py > ./selection/compare_end2end_old_v2_order_imdb_reall.log  2>&1 &