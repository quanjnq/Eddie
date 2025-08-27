import os
from torch.utils.data import DataLoader
import argparse

from eddie.eddie_model import Eddie
from eddie.eddie_dataset import *
from data_process.data_process import split_dataset_by_sql_kfold, split_key_of, data_preprocess
from trainer.trainer import *
from util.draw_util import *
from util.log_util import *
from util.random_util import seed_everything
from util.metrics_util import print_k_fold_avg_scores
from feat.feat_eddie import eddie_feat_data

import logging
import util.const_util as const

class ModelArgs:
    def __init__(self):
        # training params
        self.lr = 0.0001
        self.batch_size = 16
        self.device = "cuda:0"
        self.epochs = 100
        self.clip_size = 8
        
        self.log_label = True
        self.k_folds = 5
        
        self.max_sort_col_num = 5
        self.max_output_col_num = 5
        self.max_predicate_num = 120
        self.max_attn_dist = 60
        
        # eddie model params
        self.disable_idx_attn = False 
    pass


def main(run_cfg):
    seed_everything(0)
    
    model_args = ModelArgs()
    
    run_id = run_cfg["run_id"]
    model_name = run_cfg["model_name"]
    dataset_path = run_cfg["dataset_path"]
    db_stat_path = run_cfg["db_stat_path"]
    checkpoints_path = run_cfg["checkpoints_path"]
    clip_label = run_cfg["clip_label"] == "True"
    ablation_feature = run_cfg["ablation_feature"] if "ablation_feature" in run_cfg else ""
    
    
    model_args.disable_idx_attn = ablation_feature == const.WO_ATTN

    setup_logging(run_id)
    logging.info(f"ablation_feature: {ablation_feature}")
    logging.info(f"using run_id: {run_id}, model_name: {model_name}")
    logging.info(f"loading dataset from {dataset_path}")
    logging.info(f"loading db_stat_path from {db_stat_path}")
    logging.info(f"checkpoints_path: {checkpoints_path}")
    
    data_items, db_stat = data_preprocess(dataset_path, db_stat_path, model_args.log_label, clip_label=clip_label)
    data_items = eddie_feat_data(data_items, db_stat, enable_histogram=ablation_feature != const.WO_HIST)
    
    vary_eval = False
    if run_cfg.get('vary_dataset_path') or run_cfg.get('vary_db_stat_path') or run_cfg.get('vary_schema'):
        vary_eval = True
        logging.info(f"is child exp, load new eval dataset")
        logging.info(f"[new eval dataset] dataset from {run_cfg['vary_dataset_path']}")
        logging.info(f"[new eval dataset] db_stat_path from {run_cfg['vary_db_stat_path']}")
        
        vary_data_items, vary_db_stat = data_preprocess(run_cfg["vary_dataset_path"], run_cfg["vary_db_stat_path"], model_args.log_label,
                                                      random_change_tbl_col=run_cfg["vary_schema"], clip_label=clip_label)
        vary_data_items = eddie_feat_data(vary_data_items, vary_db_stat)
        
    train_scores_list = []
    val_scores_list = []
    vary_val_scores_list = []

    os.makedirs('./images', exist_ok=True)
    img_path = f'./images/{run_id}.jpg'
    fig, axes = init_fold_plot(model_args.k_folds)
    
    # k-fold cross validation
    k_folds_datasets = split_dataset_by_sql_kfold(data_items, model_args.k_folds)
    for fold_i, (train_items, val_items) in enumerate(k_folds_datasets):
        logging.info(f"**************************** Fold-{fold_i} Start ****************************")
        
        model = Eddie(max_sort_col_num=model_args.max_sort_col_num, max_output_col_num=model_args.max_output_col_num, max_predicate_num=model_args.max_predicate_num, \
                            disable_idx_attn=model_args.disable_idx_attn, clip_label=clip_label, ablation_feat=ablation_feature)

        os.makedirs(checkpoints_path, exist_ok=True)
        model_path = f"{checkpoints_path}/fold_{fold_i}.pth"
        
        if not os.path.exists(model_path):
            logging.info(f"{model_path} no exist, start training")
            train_ds = EddieDataset(train_items)
            val_ds = EddieDataset(val_items)
            logging.info(f"len train data {len(train_ds)}")
            logging.info(f"len val data {len(val_ds)}")
        
            dataCollator = EddieDataCollator(model_args.max_sort_col_num, model_args.max_output_col_num, model_args.max_attn_dist, model_args.log_label)
            train_dataloader = DataLoader(train_ds, batch_size=model_args.batch_size, collate_fn=dataCollator.collate_fn, shuffle=True, num_workers=16, pin_memory=True)
            val_dataloader = DataLoader(val_ds, batch_size=model_args.batch_size, collate_fn=dataCollator.collate_fn, num_workers=16, pin_memory=True)
            
            train_loss_list, train_pred_list, train_label_list, train_scores, \
            val_loss_list, val_pred_list, val_label_list, val_scores = train(model, train_dataloader, val_dataloader, model_args, model_save_path=model_path)
            
            train_scores_list.append(train_scores)
            val_scores_list.append(val_scores)
            
            update_fold_plot(img_path, fig, axes, fold_i, train_loss_list, val_loss_list, train_label_list, train_pred_list, val_label_list, val_pred_list)
            
        if vary_eval:
            logging.info("load checkpoints and eval new dataset")
            val_keys = set([split_key_of(item) for item in val_items])
            vary_val_items = [item for item in vary_data_items if split_key_of(item) in val_keys]
        
            val_ds = EddieDataset(vary_val_items)
            logging.info(f"len val data {len(val_ds)}")
            dataCollator = EddieDataCollator(model_args.max_sort_col_num, model_args.max_output_col_num, model_args.max_attn_dist, model_args.log_label)
            val_dataloader = DataLoader(val_ds, batch_size=model_args.batch_size, collate_fn=dataCollator.collate_fn, num_workers=16, pin_memory=True)
    
            logging.info(f"load model from {model_path}")
            checkpoint = torch.load(model_path, map_location="cpu")
            model.load_state_dict(checkpoint["model"])
            model.to(model_args.device)

            logging.info("start infering")
            val_loss_list, val_pred_list, val_label_list, val_scores = evaluate(model, val_dataloader, model_args.device, model_args)
            vary_val_scores_list.append(val_scores)
        
        logging.info(f"**************************** Fold-{fold_i} End ****************************\n\n")
    
    if len(train_scores_list) > 0:
        print_k_fold_avg_scores(train_scores_list, val_scores_list)
    if len(vary_val_scores_list) > 0:
        print_k_fold_avg_scores(train_scores_list, vary_val_scores_list)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run Eddie model training and evaluation')
    
    # Required arguments
    parser.add_argument('--run_id', type=str, required=True, help=' Unique identifier for this run. This will be used for logging and checkpoint naming.')
    parser.add_argument('--model_name', type=str, required=True, help='Name of the model to be used.')
    parser.add_argument('--dataset_path', type=str, required=True, help='Path to the dataset used for K-fold cross-validation.')
    parser.add_argument('--db_stat_path', type=str, required=True, help='Path to the database statistics file.')
    parser.add_argument('--checkpoints_path', type=str, required=True, help='Directory to save or load model checkpoints. If the specified checkpoint does not exist, the model will be trained during K-fold cross-validation.')
    
    # Optional arguments
    parser.add_argument('--vary_dataset_path', type=str, help='Path to a dataset for testing drift scenarios (e.g., changes in queries or data statistics)')
    parser.add_argument('--vary_db_stat_path', type=str, help='Path to alternative database statistics, used when data statistics change in drift scenarios.')
    parser.add_argument('--vary_schema', type=bool, default=False, help='Whether to vary the schema of the dataset.')
    parser.add_argument('--clip_label', type=str, default="True", help='Whether to limit the training label interval to 0 - 1. True or False')
    
    args = parser.parse_args()
    run_cfg = vars(args)  # Convert args directly to dictionary
    
    main(run_cfg)

''' example
nohup python run_eddie.py \
    --run_id tpcds__base_w_init_idx__eddie_v1_0 \
    --model_name eddie \
    --dataset_path ./datasets/tpcds__base_wo_init_idx.pickle \
    --db_stat_path ./db_stats_data/indexselection_tpcds___10_stats.json \
    --checkpoints_path ./checkpoints/tpcds__base_w_init_idx__eddie_v1_0 \
    > /dev/null 2>&1 &
'''
