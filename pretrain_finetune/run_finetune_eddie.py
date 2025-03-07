import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from torch.utils.data import DataLoader

from eddie.eddie_model import Eddie
from eddie.eddie_dataset import *
from data_process.data_process import split_dataset_by_sql_kfold
from trainer.trainer import *
from util.draw_util import *
from util.log_util import *
from util.random_util import seed_everything
from util.string_util import *
from util.metrics_util import print_k_fold_avg_scores
from feat.feat_eddie import eddie_feat_data
import random
import logging
from pretrain_finetune.pretrain_finetune_config import *
from data_process.data_process import data_preprocess
from run_eddie import ModelArgs
import argparse

def main(run_cfg):
    seed_everything(0)
    
    args = ModelArgs()
    
    workload_name = run_cfg["workload_name"]
    finetune_data_rate = float(run_cfg["finetune_data_rate"])
    use_pretrain_model = run_cfg["use_pretrain_model"] if "use_pretrain_model" in run_cfg else True # Initialize with pretrained model parameters
    checkpoints_path = run_cfg["checkpoints_path"]
    model_name = run_cfg["model_name"]
    
    run_id = run_cfg["run_id"]
    if "parent_run_id" in run_cfg:
        parent_run_id = run_cfg["parent_run_id"]
        pre_train_checkpoints_path = f"./checkpoints/{parent_run_id}/{workload_name}__{model_name}.pth"
    else:
        pre_train_checkpoints_path = run_cfg["pretrain_model_path"] if "pretrain_model_path" in run_cfg else None
    
    finetune_dataset_path = pretrain_finetune_config[workload_name]["finetune_dataset_path"]
    
    setup_logging(run_id)
    
    db_stat_path = pretrain_finetune_config[workload_name]["db_stat_path"]
    logging.info(f"loading finetune dataset from {finetune_dataset_path}")
    logging.info(f"using run_id {run_id}")
    
    
    data_items, db_stat = data_preprocess(finetune_dataset_path, db_stat_path, args.log_label)
    data_items = eddie_feat_data(data_items, db_stat)
    
    train_scores_list = []
    val_scores_list = []

    img_path = f'./images/{run_id}.jpg'
    fig, axes = init_fold_plot(args.k_folds)
    
    os.makedirs(checkpoints_path, exist_ok=True)
    
    # k-fold cross validation
    k_folds_datasets = split_dataset_by_sql_kfold(data_items, args.k_folds)
    for fold_i, (train_items, val_items) in enumerate(k_folds_datasets):
        logging.info(f"**************************** Fold-{fold_i} Start ****************************")
        random.shuffle(train_items)
        train_items = train_items[:int(len(train_items)*finetune_data_rate)]
        train_ds = EddieDataset(train_items)
        val_ds = EddieDataset(val_items)
        logging.info(f"len train data {len(train_ds)}")
        logging.info(f"len val data {len(val_ds)}")

        dataCollator = EddieDataCollator(args.max_sort_col_num, args.max_output_col_num, args.max_attn_dist, args.log_label)
        train_dataloader = DataLoader(train_ds, batch_size=args.batch_size, collate_fn=dataCollator.collate_fn, shuffle=True, num_workers=16, pin_memory=True)
        val_dataloader = DataLoader(val_ds, batch_size=args.batch_size, collate_fn=dataCollator.collate_fn, num_workers=16, pin_memory=True)
        # load pretrain
        model = Eddie(max_sort_col_num=args.max_sort_col_num, max_output_col_num=args.max_output_col_num, max_predicate_num=args.max_predicate_num, \
                        disable_idx_attn=args.disable_idx_attn)
        
        if use_pretrain_model:
            logging.info(f"load pretrain model from {pre_train_checkpoints_path}")
            checkpoint = torch.load(pre_train_checkpoints_path, map_location="cpu")
            model.load_state_dict(checkpoint["model"])
            
        model.to(args.device)
        
        logging.info("start training")
        model_save_path = f"{checkpoints_path}/fold_{fold_i}.pth"
        train_loss_list, train_pred_list, train_label_list, train_scores, \
           val_loss_list, val_pred_list, val_label_list, val_scores = train(model, train_dataloader, val_dataloader, args, model_save_path=model_save_path)
        
        train_scores_list.append(train_scores)
        val_scores_list.append(val_scores)
        
        update_fold_plot(img_path, fig, axes, fold_i, train_loss_list, val_loss_list, train_label_list, train_pred_list, val_label_list, val_pred_list)
        logging.info(f"**************************** Fold-{fold_i} End ****************************\n\n")

    
    print_k_fold_avg_scores(train_scores_list, val_scores_list)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run Eddie model fine-tuning')
    
    # Required arguments
    parser.add_argument('--run_id', type=str, required=True, help=' Unique identifier for this run. This will be used for logging and checkpoint naming.')
    parser.add_argument('--model_name', type=str, required=True, help='Name of the model to be used.')
    parser.add_argument('--checkpoints_path', type=str, required=True, help='Directory to save or load model checkpoints.')
    parser.add_argument('--workload_name', type=str, required=True, help='This parameter is used to search for the specific experimental parameters of the configuration file in the Step 1')
    
    # Optional arguments
    parser.add_argument('--pretrain_model_path', type=str, help='Used to find pre-trained model that have been saved. Parameters used for initializing fine-tuning model')
    parser.add_argument('--finetune_data_rate', type=str, default="0.5", help='Fine-tune the target data ratio of the model, with a value range of 0-1')
    
    args = parser.parse_args()
    cfg = vars(args)  # Convert args directly to dictionary
    main(cfg)
    
''' example
python pretrain_finetune/run_finetune_eddie.py \
    --run_id tpcds__finetune \
    --model_name eddie \
    --checkpoints_path ./checkpoints/finetune_tpcds \
    --workload_name tpcds \
    --pretrain_model_path ./checkpoints/pretrain_tpcds/tpcds__eddie.pth \
    --finetune_data_rate 0.5
'''