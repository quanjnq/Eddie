import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from torch.utils.data import DataLoader

from eddie.eddie_model import Eddie
from eddie.eddie_dataset import *
from trainer.trainer import *
from util.draw_util import *
from util.log_util import *
from util.random_util import seed_everything
from util.string_util import *
from util.metrics_util import print_k_fold_avg_scores
from feat.feat_eddie import eddie_feat_data
from pretrain_finetune.pretrain_finetune_config import *
from data_process.data_process import data_preprocess
from run_eddie import ModelArgs
import argparse

import logging

def main(run_cfg):
        
    seed_everything(0)
    args = ModelArgs()
        
    run_id = run_cfg["run_id"]
    workload_name = run_cfg["workload_name"]
    checkpoints_path = run_cfg["checkpoints_path"]
    model_name = run_cfg["model_name"]
    
    pretrain_workloads = pretrain_finetune_config[workload_name]["pretrain_workloads"]
    used_pretrain_datasets_dict = {}
    for w_name in pretrain_workloads:
        used_pretrain_datasets_dict[w_name] = pretrain_all_datasets_dict[w_name]

    
    setup_logging(run_id)
    
    logging.info(f"loading pretrain_data_list_dict from {used_pretrain_datasets_dict}")
    logging.info(f"using run_id {run_id}")
        
    pretrain_data_items = []
    for dataset_type in used_pretrain_datasets_dict:
        data_list = used_pretrain_datasets_dict[dataset_type]
        if len(data_list) == 0:
            continue
        for ds_fp in data_list:
            db_stat_fp = db_stats_dict[dataset_type]
            data_items, db_stat = data_preprocess(ds_fp, db_stat_fp, args.log_label)
            data_items = eddie_feat_data(data_items, db_stat, max_sample_cnt=10000)
            pretrain_data_items.extend(data_items)
            
    train_scores_list = []
    val_scores_list = []
    gruop_scores_list = []

    img_path = f'./images/{run_id}.jpg'
    fig, axes = init_fold_plot(args.k_folds)
    
    # k-fold cross validation
    train_ds = EddieDataset(pretrain_data_items)
    mock_val_items = pretrain_data_items[:10]
    val_ds = EddieDataset(mock_val_items)
    logging.info(f"len train data {len(train_ds)}")
    logging.info(f"len val data {len(val_ds)}")

    dataCollator = EddieDataCollator(args.max_sort_col_num, args.max_output_col_num, args.max_attn_dist, args.log_label)
    train_dataloader = DataLoader(train_ds, batch_size=args.batch_size, collate_fn=dataCollator.collate_fn, shuffle=True, num_workers=16, pin_memory=True)
    val_dataloader = DataLoader(val_ds, batch_size=args.batch_size, collate_fn=dataCollator.collate_fn, num_workers=16, pin_memory=True)
    model = Eddie(max_sort_col_num=args.max_sort_col_num, max_output_col_num=args.max_output_col_num, max_predicate_num=args.max_predicate_num, \
                        disable_idx_attn=args.disable_idx_attn)
    os.makedirs(checkpoints_path, exist_ok=True)
    model_save_path = f"{checkpoints_path}/{workload_name}__{model_name}.pth"
    
    logging.info("start training")
    train_loss_list, train_pred_list, train_label_list, train_scores, \
        val_loss_list, val_pred_list, val_label_list, val_scores = train(model, train_dataloader, val_dataloader, args, model_save_path=model_save_path)
    
    train_scores_list.append(train_scores)
    val_scores_list.append(val_scores)
    
    update_fold_plot(img_path, fig, axes, 0, train_loss_list, val_loss_list, train_label_list, train_pred_list, val_label_list, val_pred_list)    
    print_k_fold_avg_scores(train_scores_list, val_scores_list, gruop_scores_list)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run Eddie model training and evaluation')
    
    # Required arguments
    parser.add_argument('--run_id', type=str, required=True, help=' Unique identifier for this run. This will be used for logging and checkpoint naming.')
    parser.add_argument('--model_name', type=str, required=True, help='Name of the model to be used.')
    parser.add_argument('--checkpoints_path', type=str, required=True, help='Directory to save or load model checkpoints.')
    parser.add_argument('--workload_name', type=str, required=True, help='This parameter is used to search for the specific experimental parameters of the configuration file in the Step 1')
    
    args = parser.parse_args()
    cfg = vars(args)  # Convert args directly to dictionary
    
    main(cfg)

''' example
python pretrain_finetune/run_pretrain_eddie.py \
    --run_id pretrain_tpcds \
    --model_name eddie \
    --checkpoints_path ./checkpoints/pretrain_tpcds \
    --workload_name tpcds
'''
