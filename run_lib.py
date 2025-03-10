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


class ModelArgs:
    def __init__(self):
        # training params
        self.lr = 0.001
        self.batch_size = 20
        self.device = "cuda:0"
        self.epochs = 100
        self.clip_size = None
        
        self.log_label = False
        self.k_folds = 5
        
        # lib model params
        self.dim1 = 32 # embedding size
        self.dim2 = 64 # hidden dimension for prediction layer
        self.dim3 = 128 # hidden dimension for FNN
        self.n_encoder_layers = 6 # number of layer of attention encoder
        self.n_heads = 8 # number of heads in attention
        self.dropout_r = 0.2 # dropout ratio
    pass

def main(run_cfg):
    seed_everything(0)
    
    args = ModelArgs()
    
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
        
    vary_eval = False
    if run_cfg.get('vary_dataset_path') or run_cfg.get('vary_db_stat_path') or run_cfg.get('vary_schema'):
        vary_eval = True
        logging.info(f"is child exp, load new eval dataset")
        logging.info(f"[new eval dataset] dataset from {run_cfg['vary_dataset_path']}")
        logging.info(f"[new eval dataset] db_stat_path from {run_cfg['vary_db_stat_path']}")
        
        vary_data_items, vary_db_stat = data_preprocess(run_cfg["vary_dataset_path"], run_cfg["vary_db_stat_path"], args.log_label,
                                                        random_change_tbl_col=run_cfg["vary_schema"])
        
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
        
        # lib model
        encoder_model, pooling_model = make_model(args.dim1, args.n_encoder_layers, args.dim3, args.n_heads, dropout=args.dropout_r)
        model = self_attn_model(encoder_model, pooling_model, 12, args.dim1, args.dim2)

        os.makedirs(checkpoints_path, exist_ok=True)
        model_path = f"{checkpoints_path}/fold_{fold_i}.pth"
        
        if not os.path.exists(model_path):
            train_ds = LibDataset(train_items, db_stat)
            val_ds = LibDataset(val_items, db_stat)
            logging.info(f"len train data {len(train_ds)}")
            logging.info(f"len val data {len(val_ds)}")
        
            train_dataloader = DataLoader(train_ds, batch_size=args.batch_size, collate_fn=collate_fn4lib, shuffle=True, num_workers=16, pin_memory=True)
            val_dataloader = DataLoader(val_ds, batch_size=args.batch_size, collate_fn=collate_fn4lib, num_workers=16, pin_memory=True)
            
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
        
            val_ds = LibDataset(vary_val_items, vary_db_stat)
            logging.info(f"len val data {len(val_ds)}")
            val_dataloader = DataLoader(val_ds, batch_size=args.batch_size, collate_fn=collate_fn4lib, num_workers=16, pin_memory=True)
    
            logging.info(f"load model from {model_path}")
            checkpoint = torch.load(model_path, map_location="cpu")
            model.load_state_dict(checkpoint["model"])
            model.to(args.device)

            logging.info("start infering")
            val_loss_list, val_pred_list, val_label_list, val_scores = evaluate(model, val_dataloader, args.device, args)
            vary_val_scores_list.append(val_scores)
        
        logging.info(f"**************************** Fold-{fold_i} End ****************************\n\n")
    
    if len(train_scores_list) > 0:
        print_k_fold_avg_scores(train_scores_list, val_scores_list)
    if len(vary_val_scores_list) > 0:
        print_k_fold_avg_scores(train_scores_list, vary_val_scores_list)

if __name__ == '__main__':
    run_cfg = {
        'run_id': 'tpcds__base_w_init_idx__lib_v1', 
        'model_name': 'lib', 
        'checkpoints_path': './checkpoints/tpcds__base_w_init_idx__lib_v1', 
        'dataset_path': './datasets/tpcds__base_w_init_idx.pickle', 
        'db_stat_path': './db_stats_data/indexselection_tpcds___10_stats.json'
    }
    main(run_cfg)
