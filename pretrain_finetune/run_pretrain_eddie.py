import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
from torch.utils.data import DataLoader

from eddie.eddie_model import Eddie
from eddie.eddie_dataset import *
from trainer.trainer import *
from util.draw_util import *
from util.log_util import *
from util.random_util import seed_everything
from util.string_util import *
from util.metrics_util import print_k_fold_avg_scores
from feat.feat_eddie import feat_data
from pretrain_finetune.pretrain_dataset_config import tpcds_pretrains
from data_process.data_process import data_preprocess

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
        self.simple_predicate_encode = False
        # self.dropout = 0.1
        # self.embed_size = 64
        # self.ff_dim = 128
        # self.head_size = 12
        # self.n_layers = 8
        # self.pred_hid = 128
        
        # dataset params
        self.enable_template_split = False
        
    pass



if __name__ == '__main__':
    seed_everything(0)
    
    args = Args()
    TPCDS = "tpcds"
    TPCH = "tpch"
    IMDB = "imdb"
    AIRLINE = "airline"
    SSB = "ssb"
    WALMART = "walmart"
    ACCIDENTS = "accidents"
    
    TOURNAMENT = "tournament"
    BASKETBALL = "basketball"
    CARCINOGENESIS = "carcinogenesis"
    CONSUMER = "consumer"
    CREDIT = "credit"
    FHNK = "fhnk"
    EMPLOYEE = "employee"
    GENOME = "genome"
    HEPATITIS = "hepatitis"
    MOVIELENS = "movielens"
    FINANCIAL =  "financial"
    BASEBALL =   "baseball"
    GENEEA   =   "geneea"
    SEZNAM  =    "seznam"
    
    TOURNAMENT_SCALED = "tournament_scaled"
    BASKETBALL_SCALED = "basketball_scaled"
    CARCINOGENESIS_SCALED = "carcinogenesis_scaled"
    CONSUMER_SCALED = "consumer_scaled"
    CREDIT_SCALED = "credit_scaled"
    FHNK_SCALED = "fhnk_scaled"
    EMPLOYEE_SCALED = "employee_scaled"
    GENOME_SCALED = "genome_scaled"
    HEPATITIS_SCALED = "hepatitis_scaled"
    MOVIELENS_SCALED = "movielens_scaled"
    FINANCIAL_SCALED =  "financial_scaled"
    BASEBALL_SCALED =   "baseball_scaled"
    GENEEA_SCALED   =   "geneea_scaled"
    SEZNAM_SCALED  =    "seznam_scaled"
    
    pretrain_data_list_dict_all = {

                        TPCDS: ["./dataset/tpcds.pickle"],
                        TPCH: ["./dataset/tpch.pickle"],
                          IMDB: ["./dataset/imdb.pickle",],

                          AIRLINE: ["./dataset/airline.pickle",],
                          SSB: ["./dataset/ssb.pickle", ],
                          WALMART: ["./dataset/walmart.pickle"],
                          ACCIDENTS: ["./dataset/accidents.pickle",],
                          
                        TOURNAMENT: ["./dataset/tournament.pickle", ],
                        BASKETBALL: ["./dataset/basketball.pickle", ],
                        CARCINOGENESIS: ["./dataset/carcinogenesis.pickle",],
                        CONSUMER: ["./dataset/consumer.pickle", ],
                        CREDIT: ["./dataset/credit.pickle"],
                        FHNK: ["./dataset/fhnk.pickle", ],
                        EMPLOYEE: ["./dataset/employee.pickle", ],
                        GENOME: ["./dataset/genome.pickle", ],
                        HEPATITIS: ["./dataset/hepatitis.pickle", ],
                        MOVIELENS: ["./dataset/movielens.pickle", ],
                        FINANCIAL: ["./dataset/financial.pickle",],
                        BASEBALL: ["./dataset/baseball.pickle", ],
                        GENEEA: ["./dataset/geneea.pickle", ],
                        SEZNAM: ["./dataset/seznam.pickle", ],
                        
                        TOURNAMENT_SCALED: ["./dataset/tournament_scaled.pickle", ],
                        BASKETBALL_SCALED: ["./dataset/basketball_scaled.pickle", ],
                        CARCINOGENESIS_SCALED: ["./dataset/carcinogenesis_scaled.pickle",],
                        CONSUMER_SCALED: ["./dataset/consumer_scaled.pickle", ],
                        CREDIT_SCALED: ["./dataset/credit_scaled.pickle"],
                        FHNK_SCALED: ["./dataset/fhnk_scaled.pickle", ],
                        EMPLOYEE_SCALED: ["./dataset/employee_scaled.pickle", ],
                        GENOME_SCALED: ["./dataset/genome_scaled.pickle", ],
                        HEPATITIS_SCALED: ["./dataset/hepatitis_scaled.pickle", ],
                        MOVIELENS_SCALED: ["./dataset/movielens_scaled.pickle", ],
                        FINANCIAL_SCALED: ["./dataset/financial_scaled.pickle",],
                        BASEBALL_SCALED: ["./dataset/baseball_scaled.pickle", ],
                        GENEEA_SCALED: ["./dataset/geneea_scaled.pickle", ],
                        SEZNAM_SCALED: ["./dataset/seznam_scaled.pickle", ],
                        
                          }

    db_stats = {IMDB: "./db_stats_data/imdbload_stats.json",
                TPCH: "./db_stats_data/indexselection_tpch___10_stats.json",
                TPCDS: './db_stats_data/indexselection_tpcds___10_stats.json',
                AIRLINE: './db_stats_data/airline_stats1.json',
                SSB: './db_stats_data/ssb_stats1.json',
                WALMART:         "./db_stats_data/walmart_stats.json",
                ACCIDENTS:       "./db_stats_data/accidents_stats.json",
                
                TOURNAMENT:      "./db_stats_data/tournament_stats.json",
                BASKETBALL:      "./db_stats_data/basketball_stats.json",
                CARCINOGENESIS:  "./db_stats_data/carcinogenesis_stats.json",
                CONSUMER:        "./db_stats_data/consumer_stats.json",
                CREDIT:          "./db_stats_data/credit_stats.json",
                FHNK:            "./db_stats_data/fhnk_stats.json",
                EMPLOYEE:        "./db_stats_data/employee_stats.json",
                GENOME:          "./db_stats_data/genome_stats.json",
                HEPATITIS:       "./db_stats_data/hepatitis_stats.json",
                MOVIELENS:       "./db_stats_data/movielens_stats.json",
                FINANCIAL:       "./db_stats_data/financial_stats.json",
                BASEBALL:       "./db_stats_data/baseball_stats.json",
                GENEEA:       "./db_stats_data/geneea_stats.json",
                SEZNAM:       "./db_stats_data/seznam_stats.json",
                
                TOURNAMENT_SCALED:      "./db_stats_data/tournament_scaled_stats.json",
                BASKETBALL_SCALED:      "./db_stats_data/basketball_scaled_stats.json",
                CARCINOGENESIS_SCALED:  "./db_stats_data/carcinogenesis_scaled_stats.json",
                CONSUMER_SCALED:        "./db_stats_data/consumer_scaled_stats.json",
                CREDIT_SCALED:          "./db_stats_data/credit_scaled_stats.json",
                FHNK_SCALED:            "./db_stats_data/fhnk_scaled_stats.json",
                EMPLOYEE_SCALED:        "./db_stats_data/employee_scaled_stats.json",
                GENOME_SCALED:          "./db_stats_data/genome_scaled_stats.json",
                HEPATITIS_SCALED:       "./db_stats_data/hepatitis_scaled_stats.json",
                MOVIELENS_SCALED:       "./db_stats_data/movielens_scaled_stats.json",
                FINANCIAL_SCALED:       "./db_stats_data/financial_scaled_stats.json",
                BASEBALL_SCALED:       "./db_stats_data/baseball_scaled_stats.json",
                GENEEA_SCALED:       "./db_stats_data/geneea_scaled_stats.json",
                SEZNAM_SCALED:       "./db_stats_data/seznam_scaled_stats.json",
                }
    pretrain_data_list_dict =  {}
    # pretrains = [IMDB, TPCH, ACCIDENTS,  AIRLINE ,  BASEBALL_SCALED ,  BASKETBALL_SCALED ,  CARCINOGENESIS_SCALED ,  CONSUMER_SCALED ,  CREDIT_SCALED ,  EMPLOYEE_SCALED ,  FHNK_SCALED ,  FINANCIAL_SCALED ,  GENEEA_SCALED ,  GENOME_SCALED ,  HEPATITIS_SCALED ,  MOVIELENS_SCALED ,  SEZNAM_SCALED ,  SSB ,  TOURNAMENT_SCALED ,  WALMART ]
    pretrains = tpcds_pretrains
    # pretrains = tpcds_pretrains[:1]
    # pretrains = [ ACCIDENTS,  AIRLINE ,  BASEBALL_SCALED ,  BASKETBALL_SCALED ,  CARCINOGENESIS_SCALED ,  CONSUMER_SCALED ,  CREDIT_SCALED ,  EMPLOYEE_SCALED ,  FHNK_SCALED ,  FINANCIAL_SCALED ,  GENEEA_SCALED ,  GENOME_SCALED ,  HEPATITIS_SCALED ,  MOVIELENS_SCALED ,  SEZNAM_SCALED ,  SSB ,  TOURNAMENT_SCALED ,  WALMART ]
    # pretrains = [ WALMART ]
    for db in pretrains:
        pretrain_data_list_dict[db] = pretrain_data_list_dict_all[db]
    

    exp_id = "ours_v13"
    if args.enable_template_split:
        exp_id = exp_id + "_split_tpl"
    
    dataset_name = get_pretrain_dataset_name(pretrain_data_list_dict)
    
    exp_id = dataset_name + '_' + exp_id
    setup_logging(exp_id)
    
    logging.info(f"len {len(pretrains)} \  pretrains {pretrains}")
    logging.info(f"loading pretrain_data_list_dict from {pretrain_data_list_dict}")
    logging.info(f"using exp_id {exp_id}")
    
    pretrain_data_items = []
    for dataset_type in pretrain_data_list_dict:
        data_list = pretrain_data_list_dict[dataset_type]
        if len(data_list) == 0:
            continue
        for fp in data_list:
            with open(fp, "rb") as f:
                src_data = pickle.load(f)
            db_stat_fp = db_stats[dataset_type]
            with open(db_stat_fp, "r") as f:
                db_stat = json.load(f)
            data_items = data_preprocess(src_data, args.log_label)
            # data_items = data_items[:150]
            
            dataset_type_data_items = feat_data(data_items, db_stat, max_sample_cnt=10000)
            # dataset_type_data_items = feat_data(data_items, db_stat, max_sample_cnt=10)
        
            pretrain_data_items.extend(dataset_type_data_items)
            
    
    train_scores_list = []
    val_scores_list = []
    gruop_scores_list = []

    img_path = f'./images/{exp_id}.jpg'
    fig, axes = init_fold_plot(args.k_folds)
    
    # k-fold cross validation
    # k_folds_datasets = split_dataset_by_sql_kfold(pretrain_data_items, args.k_folds, enable_template_split=args.enable_template_split)
    # for fold_i, (train_items, val_items) in enumerate(k_folds_datasets):
    # logging.info(f"**************************** Fold-{fold_i} Start ****************************")
    # logging.info(set([it['query_type']+"#"+str(it['query'].nr) for it in val_items]))
    train_ds = EddieDataset(pretrain_data_items)
    mock_val_items = pretrain_data_items[:10]
    val_ds = EddieDataset(mock_val_items)
    logging.info(f"len train data {len(train_ds)}")
    logging.info(f"len val data {len(val_ds)}")

    dataCollator = EddieDataCollator(args.max_sort_col_num, args.max_output_col_num, args.max_attn_dist, args.log_label)
    train_dataloader = DataLoader(train_ds, batch_size=args.batch_size, collate_fn=dataCollator.collate_fn, shuffle=True, num_workers=16, pin_memory=True)
    val_dataloader = DataLoader(val_ds, batch_size=args.batch_size, collate_fn=dataCollator.collate_fn, num_workers=16, pin_memory=True)
    model = Eddie(max_sort_col_num=args.max_sort_col_num, max_output_col_num=args.max_output_col_num, max_predicate_num=args.max_predicate_num, \
                        disable_idx_attn=args.disable_idx_attn, simple_predicate_encode=args.simple_predicate_encode)
    model_save_path = f"./checkpoint/{exp_id}.pth"
    
    logging.info("start training")
    train_loss_list, train_pred_list, train_label_list, train_scores, \
        val_loss_list, val_pred_list, val_label_list, val_scores = train(model, train_dataloader, val_dataloader, args, model_save_path=model_save_path)
    
    train_scores_list.append(train_scores)
    val_scores_list.append(val_scores)
    
    update_fold_plot(img_path, fig, axes, 0, train_loss_list, val_loss_list, train_label_list, train_pred_list, val_label_list, val_pred_list)
    # logging.info(f"**************************** Fold-{fold_i} End ****************************\n\n")
        # break
    
    print_k_fold_avg_scores(train_scores_list, val_scores_list, gruop_scores_list)

#  nohup python ./pretrain_finetune200/run_pretrain_eddie.py > ./pretrain_finetune200/run_pretrain_eddie.log 2>&1 &