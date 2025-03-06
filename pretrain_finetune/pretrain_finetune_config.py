# base
TPCDS = "tpcds"
TPCH = "tpch"
IMDB = "imdb"

# ext
ACCIDENTS = "accidents"
AIRLINE = "airline"
BASEBALL = "baseball"
BASKETBALL = "basketball"
CARCINOGENESIS = "carcinogenesis"
CONSUMER = "consumer"
CREDIT = "credit"
EMPLOYEE = "employee"
FHNK = "fhnk"
FINANCIAL = "financial"
GENEEA = "geneea"
GENOME = "genome"
HEPATITIS = "hepatitis"
MOVIELENS = "movielens"
SEZNAM = "seznam"
SSB = "ssb"
TOURNAMENT = "tournament"
WALMART = "walmart"

# pre_train workload
pretrain_ext_workloads = [ACCIDENTS, AIRLINE, BASEBALL, BASKETBALL , CARCINOGENESIS , CONSUMER , CREDIT , EMPLOYEE , FHNK , FINANCIAL , GENEEA, GENOME, HEPATITIS, MOVIELENS, SEZNAM , SSB, TOURNAMENT, WALMART]
pretrain_workloads_tpcds = [IMDB, TPCH, ] + pretrain_ext_workloads
pretrain_workloads_tpch = [IMDB, TPCDS, ] + pretrain_ext_workloads
pretrain_workloads_imdb = [TPCDS, TPCH, ] + pretrain_ext_workloads


# pre_train all datasets
pretrain_all_datasets_dict = {
                        TPCDS: ["./datasets/tpcds__base_wo_init_idx.pickle"],
                        TPCH: ["./datasets/tpch__base_wo_init_idx.pickle"],
                        IMDB: ["./datasets/imdb__base_wo_init_idx.pickle",],
                        
                        ACCIDENTS: ["./datasets/pretrain_accidents.pickle",],
                        AIRLINE: ["./datasets/pretrain_airline.pickle",],
                        BASEBALL: ["./datasets/pretrain_baseball.pickle",],
                        BASKETBALL: ["./datasets/pretrain_basketball.pickle",],
                        CARCINOGENESIS: ["./datasets/pretrain_carcinogenesis.pickle",],
                        CONSUMER: ["./datasets/pretrain_consumer.pickle",],
                        CREDIT: ["./datasets/pretrain_credit.pickle",],
                        EMPLOYEE: ["./datasets/pretrain_employee.pickle",],
                        FHNK: ["./datasets/pretrain_fhnk.pickle",],
                        FINANCIAL: ["./datasets/pretrain_financial.pickle",],
                        GENEEA: ["./datasets/pretrain_geneea.pickle",],
                        GENOME: ["./datasets/pretrain_genome.pickle",],
                        HEPATITIS: ["./datasets/pretrain_hepatitis.pickle",],
                        MOVIELENS: ["./datasets/pretrain_movielens.pickle",],
                        SEZNAM: ["./datasets/pretrain_seznam.pickle",],
                        SSB: ["./datasets/pretrain_ssb.pickle",],
                        TOURNAMENT: ["./datasets/pretrain_tournament.pickle",],
                        WALMART: ["./datasets/pretrain_walmart.pickle",],
                    }

# db_stats
db_stats_dict = {
                TPCDS: "./db_stats_data/indexselection_tpcds___10_stats.json",
                TPCH: "./db_stats_data/indexselection_tpch___10_stats.json",
                IMDB: "./db_stats_data/imdbload_stats.json",
                
                ACCIDENTS: "./db_stats_data/accidents_stats.json",
                AIRLINE: "./db_stats_data/airline_stats.json",
                BASEBALL: "./db_stats_data/baseball_stats.json",
                BASKETBALL: "./db_stats_data/basketball_stats.json",
                CARCINOGENESIS: "./db_stats_data/carcinogenesis_stats.json",
                CONSUMER: "./db_stats_data/consumer_stats.json",
                CREDIT: "./db_stats_data/credit_stats.json",
                EMPLOYEE: "./db_stats_data/employee_stats.json",
                FHNK: "./db_stats_data/fhnk_stats.json",
                FINANCIAL: "./db_stats_data/financial_stats.json",
                GENEEA: "./db_stats_data/geneea_stats.json",
                GENOME: "./db_stats_data/genome_stats.json",
                HEPATITIS: "./db_stats_data/hepatitis_stats.json",
                MOVIELENS: "./db_stats_data/movielens_stats.json",
                SEZNAM: "./db_stats_data/seznam_stats.json",
                SSB: "./db_stats_data/ssb_stats.json",
                TOURNAMENT: "./db_stats_data/tournament_stats.json",
                WALMART: "./db_stats_data/walmart_stats.json",
            }


# config
pretrain_finetune_config =  {
    TPCDS: {"pretrain_workloads": pretrain_workloads_tpcds, "finetune_dataset_path": "./datasets/tpcds__base_wo_init_idx.pickle", "db_stat_path": db_stats_dict[TPCDS]},
    TPCH: {"pretrain_workloads": pretrain_workloads_tpch, "finetune_dataset_path": "./datasets/tpch__base_wo_init_idx.pickle", "db_stat_path":db_stats_dict[TPCH]},
    IMDB: {"pretrain_workloads": pretrain_workloads_imdb, "finetune_dataset_path": "./datasets/imdb__base_wo_init_idx.pickle", "db_stat_path":db_stats_dict[IMDB]},
}
