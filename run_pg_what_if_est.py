import json
import pickle
from torch.utils.data import DataLoader

from lib_est.lib_model import *
from lib_est.lib_dataset import *
from data_process.data_process import *
from trainer.trainer import *
from util.draw_util import *
from util.log_util import *
from util.random_util import seed_everything
from util.metrics_util import print_k_fold_avg_scores

import logging
from selection.dbms.postgres_dbms import PostgresDatabaseConnector


def clear_all_hypo_indexes(db_connector=None):
    stmt = "SELECT * FROM hypopg_reset();"
    db_connector.exec_only(stmt)


def create_hypo_indexes(indexes, db_connector=None):
    for index in indexes:
        table = index[0].split('.')[0]
        cols = ','.join([col.split('.')[-1] for col in index])
        statement = f"create index on {table} ({cols})"
        statement = f"select * from hypopg_create_index('{statement}')"
        db_connector.exec_only(statement)
        db_connector.commit()


class Args:
    def __init__(self):
        self.k_folds = 5
        self.use_virtual_index = True
    pass


def main(run_cfg):
    seed_everything(0)
    
    args = Args()
    
    run_id = run_cfg["run_id"]
    model_name = run_cfg["model_name"]
    dataset_path = run_cfg["dataset_path"]
    db_stat_path = run_cfg["db_stat_path"]

    setup_logging(run_id)
    logging.info(f"using run_id: {run_id}, model_name: {model_name}")
    logging.info(f"loading dataset from {dataset_path}")
    logging.info(f"loading db_stat_path from {db_stat_path}")
    
    data_items, db_stat = data_preprocess(dataset_path, db_stat_path)
        
    vary_eval = False
    if run_cfg.get('vary_dataset_path') or run_cfg.get('vary_db_stat_path') or run_cfg.get('vary_schema'):
        vary_eval = True
        logging.info(f"is child exp, load new eval dataset")
        logging.info(f"[new eval dataset] dataset from {run_cfg['vary_dataset_path']}")
        logging.info(f"[new eval dataset] db_stat_path from {run_cfg['vary_db_stat_path']}")
        
        vary_data_items, vary_db_stat = data_preprocess(run_cfg["vary_dataset_path"], run_cfg["vary_db_stat_path"],
                                                      random_change_tbl_col=run_cfg["vary_schema"])
    
    database_name = run_cfg["vary_db_name"] if 'vary_db_name' in run_cfg else run_cfg["db_name"]
    logging.info(f"using database_name {database_name}")
    try:
        db_connector = PostgresDatabaseConnector(database_name, True)
    except Exception as e:
        logging.info(f"except: To connect local database {database_name} error; exception: {e}")
        return
    
    train_scores_list = []
    val_scores_list = []

    os.makedirs('./images', exist_ok=True)
    img_path = f'./images/{run_id}.jpg'
    fig, axes = init_fold_plot(args.k_folds)
    
    # k-fold cross validation
    k_folds_datasets = split_dataset_by_sql_kfold(data_items, args.k_folds)
    for fold_i, (train_items, val_items) in enumerate(k_folds_datasets):
        logging.info(f"**************************** Fold-{fold_i} Start ****************************")
        
        logging.info(f"len train data {len(train_items)}")
        logging.info(f"len val data {len(val_items)}")

        val_label_list = []
        val_pred_list = []
        for item in val_items:
            val_label_list.append(item['label'])
            
            orig_cost = item['plan_tree']['Total Cost']
            with_index_cost = item['with_index_plan']['Total Cost']
            
            if args.use_virtual_index:
                clear_all_hypo_indexes(db_connector)
                create_hypo_indexes(item['indexes'], db_connector)
                with_index_cost, with_index_plan = db_connector.exec_explain_query(item['query'])
                
            pred = (orig_cost - with_index_cost) / max(orig_cost, 1e-6)
            pred = max(pred, 0)
            val_pred_list.append(pred)
            
        val_scores = calc_qerror(val_pred_list, val_label_list)
        
        val_scores_list.append(val_scores)
        
        update_fold_plot(img_path, fig, axes, fold_i, [], [], [], [], val_label_list, val_pred_list)
        logging.info(f"**************************** Fold-{fold_i} End ****************************\n\n")
    
    print_k_fold_avg_scores(train_scores_list, val_scores_list)
    

if __name__ == '__main__':
    run_cfg = {
        'run_id': 'tpcds__base_w_init_idx__postgresql_v1', 
        'model_name': 'postgresql', 
        'workload_name': 'tpcds', 
        'dataset_path': './datasets/tpcds__base_w_init_idx.pickle', 
        'db_stat_path': './db_stats_data/indexselection_tpcds___10_stats.json', 
        'db_name': 'indexselection_tpcds___10'
    }
    main(run_cfg)