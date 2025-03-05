import os
from torch.utils.data import DataLoader

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


class Args:
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
    
    args = Args()
    
    run_id = run_cfg["run_id"]
    model_name = run_cfg["model_name"]
    dataset_path = run_cfg["dataset_path"]
    db_stat_path = run_cfg["db_stat_path"]
    checkpoints_path = run_cfg["checkpoints_path"]

    setup_logging(run_id)
    logging.info(f"using run_id: {run_id}, model_name: {model_name}")
    logging.info(f"loading dataset from {dataset_path}")
    logging.info(f"loading db_stat_path from {db_stat_path}")
    logging.info(f"checkpoints_path: {checkpoints_path}")
    
    data_items, db_stat = data_preprocess(dataset_path, db_stat_path, args.log_label)
    data_items = eddie_feat_data(data_items, db_stat)
        
    is_child_exp = True if 'is_child_exp' in run_cfg and run_cfg["is_child_exp"] else False
    if is_child_exp:
        logging.info(f"is child exp, load new eval dataset")
        logging.info(f"[new eval dataset] dataset from {run_cfg['new_dataset_path']}")
        logging.info(f"[new eval dataset] db_stat_path from {run_cfg['new_db_stat_path']}")
        
        new_data_items, new_db_stat = data_preprocess(run_cfg["new_dataset_path"], run_cfg["new_db_stat_path"], args.log_label,
                                                      random_change_tbl_col=run_cfg["vary_schema"])
        new_data_items = eddie_feat_data(new_data_items, new_db_stat)
        
    train_scores_list = []
    val_scores_list = []

    os.makedirs('./images', exist_ok=True)
    img_path = f'./images/{run_id}.jpg'
    fig, axes = init_fold_plot(args.k_folds)
    
    # k-fold cross validation
    k_folds_datasets = split_dataset_by_sql_kfold(data_items, args.k_folds)
    for fold_i, (train_items, val_items) in enumerate(k_folds_datasets):
        logging.info(f"**************************** Fold-{fold_i} Start ****************************")
        
        model = Eddie(max_sort_col_num=args.max_sort_col_num, max_output_col_num=args.max_output_col_num, max_predicate_num=args.max_predicate_num, \
                            disable_idx_attn=args.disable_idx_attn)

        os.makedirs(checkpoints_path, exist_ok=True)
        model_path = f"{checkpoints_path}/fold_{fold_i}.pth"
        if not is_child_exp:
            train_ds = EddieDataset(train_items)
            val_ds = EddieDataset(val_items)
            logging.info(f"len train data {len(train_ds)}")
            logging.info(f"len val data {len(val_ds)}")
        
            dataCollator = EddieDataCollator(args.max_sort_col_num, args.max_output_col_num, args.max_attn_dist, args.log_label)
            train_dataloader = DataLoader(train_ds, batch_size=args.batch_size, collate_fn=dataCollator.collate_fn, shuffle=True, num_workers=16, pin_memory=True)
            val_dataloader = DataLoader(val_ds, batch_size=args.batch_size, collate_fn=dataCollator.collate_fn, num_workers=16, pin_memory=True)
            
            logging.info("start training")
            train_loss_list, train_pred_list, train_label_list, train_scores, \
            val_loss_list, val_pred_list, val_label_list, val_scores = train(model, train_dataloader, val_dataloader, args, model_save_path=model_path)
            
            train_scores_list.append(train_scores)
            val_scores_list.append(val_scores)
            
            update_fold_plot(img_path, fig, axes, fold_i, train_loss_list, val_loss_list, train_label_list, train_pred_list, val_label_list, val_pred_list)
        else:
            # load checkpoints and eval new dataset
            val_keys = set([split_key_of(item) for item in val_items])
            new_val_items = [item for item in new_data_items if split_key_of(item) in val_keys]
        
            val_ds = EddieDataset(new_val_items)
            logging.info(f"len val data {len(val_ds)}")
            dataCollator = EddieDataCollator(args.max_sort_col_num, args.max_output_col_num, args.max_attn_dist, args.log_label)
            val_dataloader = DataLoader(val_ds, batch_size=args.batch_size, collate_fn=dataCollator.collate_fn, num_workers=16, pin_memory=True)
    
            logging.info(f"load model from {model_path}")
            checkpoint = torch.load(model_path, map_location="cpu")
            model.load_state_dict(checkpoint["model"])
            model.to(args.device)

            logging.info("start infering")
            val_loss_list, val_pred_list, val_label_list, val_scores = evaluate(model, val_dataloader, args.device, args)
            val_scores_list.append(val_scores)
        
        logging.info(f"**************************** Fold-{fold_i} End ****************************\n\n")
    
    print_k_fold_avg_scores(train_scores_list, val_scores_list)


if __name__ == '__main__':
    run_cfg = {
        'run_id': 'tpcds__base_w_init_idx__eddie_v1', 
        'model_name': 'eddie', 
        'checkpoints_path': './checkpoints/tpcds__base_w_init_idx__eddie_v1', 
        'dataset_path': './datasets/tpcds__base_w_init_idx.pickle', 
        'db_stat_path': './db_stats_data/indexselection_tpcds___10_stats.json'
    }
    main(run_cfg)
