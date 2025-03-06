import json
import pickle
from torch.utils.data import DataLoader

from queryformer.qf_model import *
from queryformer.qf_dataset import *
from data_process.data_process import *
from trainer.trainer import *
from util.draw_util import *
from util.log_util import *
from util.random_util import seed_everything
from util.metrics_util import print_k_fold_avg_scores
from queryformer.encoding_const import tpcds10_encoding, tpch10_encoding, imdb_encoding

import pandas as pd
from warnings import simplefilter
simplefilter(action="ignore",category=FutureWarning)
simplefilter(action="ignore",category=pd.errors.SettingWithCopyWarning)


import logging
from selection.dbms.postgres_dbms import PostgresDatabaseConnector


class ModelArgs:
    def __init__(self):
        # training params
        self.lr = 0.001
        self.batch_size = 128
        self.device = "cuda:0"
        self.epochs = 100
        self.clip_size = 50
        
        self.log_label = False
        self.k_folds = 5
        self.random_change_tbl_col = False
        
        # queryformer model params
        self.embed_size = 64
        self.pred_hid = 128
        self.ffn_dim = 128
        self.head_size = 12
        self.n_layers = 8
        self.dropout = 0.1
    pass


def main(run_cfg):
    seed_everything(0)
    
    args = ModelArgs()
    schema_name2encoding = {"tpcds":tpcds10_encoding, "tpch":tpch10_encoding, "imdb":imdb_encoding}
    
    run_id = run_cfg["run_id"]
    model_name = run_cfg["model_name"]
    dataset_path = run_cfg["dataset_path"]
    db_stat_path = run_cfg["db_stat_path"]
    checkpoints_path = run_cfg["checkpoints_path"]
    encoding = schema_name2encoding[run_cfg["workload_name"]]
    hist_file_path = run_cfg["hist_file_path"]

    setup_logging(run_id)
    logging.info(f"using run_id: {run_id}, model_name: {model_name}")
    logging.info(f"loading dataset from {dataset_path}")
    logging.info(f"loading db_stat_path from {db_stat_path}")
    logging.info(f"checkpoints_path: {checkpoints_path}")
    
    data_items, db_stat = data_preprocess(dataset_path, db_stat_path, args.log_label)
    hist_file = get_hist_file(hist_file_path)
        
    vary_eval = False
    if run_cfg.get('vary_dataset_path') or run_cfg.get('vary_db_stat_path') or run_cfg.get('vary_schema'):
        vary_eval = True
        logging.info(f"is child exp, load new eval dataset")
        logging.info(f"[new eval dataset] dataset from {run_cfg['vary_dataset_path']}")
        logging.info(f"[new eval dataset] db_stat_path from {run_cfg['vary_db_stat_path']}")
        
        vary_data_items, vary_db_stat = data_preprocess(run_cfg["vary_dataset_path"], run_cfg["vary_db_stat_path"], args.log_label,
                                                      random_change_tbl_col=run_cfg["vary_schema"])
        vary_hist_file = get_hist_file(run_cfg["vary_hist_file_path"])
        
    train_scores_list = []
    val_scores_list = []
    vary_val_scores_list = []

    os.makedirs('./images', exist_ok=True)
    img_path = f'./images/{run_id}.jpg'
    fig, axes = init_fold_plot(args.k_folds)
    
    # k-fold cross validation
    k_folds_datasets = split_dataset_by_sql_kfold(data_items, args.k_folds)
    for fold_i, (train_items, val_items) in enumerate(k_folds_datasets):
        logging.info(f"**************************** Fold-{fold_i} Start ****************************")
        
        model = QueryFormerWithIndex(emb_size = args.embed_size ,ffn_dim = args.ffn_dim, head_size = args.head_size, \
                                        dropout = args.dropout, n_layers = args.n_layers, \
                                        use_sample = True, use_hist = True, \
                                        pred_hid = args.pred_hid
                                    )

        os.makedirs(checkpoints_path, exist_ok=True)
        model_path = f"{checkpoints_path}/fold_{fold_i}.pth"
        
        if not os.path.exists(model_path):
            train_ds = QfDataset(train_items, encoding, hist_file, None)
            val_ds = QfDataset(val_items, encoding, hist_file, None)
            logging.info(f"len train data {len(train_ds)}")
            logging.info(f"len val data {len(val_ds)}")
        
            train_dataloader = DataLoader(train_ds, batch_size=args.batch_size, collate_fn=collator, shuffle=True, num_workers=0, pin_memory=False)
            val_dataloader = DataLoader(val_ds, batch_size=args.batch_size, collate_fn=collator, num_workers=0, pin_memory=False)
            
            logging.info("start training")
            train_loss_list, train_pred_list, train_label_list, train_scores, \
            val_loss_list, val_pred_list, val_label_list, val_scores = train(model, train_dataloader, val_dataloader, args, model_save_path=model_path)
            
            train_scores_list.append(train_scores)
            val_scores_list.append(val_scores)
            
            update_fold_plot(img_path, fig, axes, fold_i, train_loss_list, val_loss_list, train_label_list, train_pred_list, val_label_list, val_pred_list)
        
        if vary_eval:
            logging.info("load checkpoints and eval new dataset")
            # load checkpoints and eval new dataset
            val_keys = set([split_key_of(item) for item in val_items])
            vary_val_items = [item for item in vary_data_items if split_key_of(item) in val_keys]
        
            val_ds = QfDataset(vary_val_items, encoding, vary_hist_file, None)
            logging.info(f"len val data {len(val_ds)}")
            val_dataloader = DataLoader(val_ds, batch_size=args.batch_size, collate_fn=collator, num_workers=0, pin_memory=False)
    
            logging.info(f"load model from {model_path}")
            checkpoint = torch.load(model_path, map_location="cpu")
            model.load_state_dict(checkpoint["model"])
            model.to(args.device)

            logging.info("start infering")
            val_loss_list, val_pred_list, val_label_list, val_scores = evaluate(model, val_dataloader, args.device, args)
            val_scores_list.append(val_scores)
        
        logging.info(f"**************************** Fold-{fold_i} End ****************************\n\n")
        
    if len(train_scores_list) > 0:
        print_k_fold_avg_scores(train_scores_list, val_scores_list)
    if len(vary_val_scores_list) > 0:
        print_k_fold_avg_scores(train_scores_list, vary_val_scores_list)

if __name__ == '__main__':
    run_cfg = {
        'run_id': 'tpcds__base_w_init_idx__queryformer_v1', 
        'model_name': 'queryformer', 
        'workload_name': 'tpcds', 
        'checkpoints_path': './checkpoints/tpcds__base_w_init_idx__queryformer_v1', 
        'dataset_path': './datasets/tpcds__base_w_init_idx.pickle', 
        'db_stat_path': './db_stats_data/indexselection_tpcds___10_stats.json', 
        'hist_file_path': './db_stats_data/indexselection_tpcds___10_hist_file.csv'
    }
    main(run_cfg)
