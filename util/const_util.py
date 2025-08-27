# PATH DEFINE
workload_save_path = "./workload/"
datasets_savr_dir = "./datasets/"
data_gen_work_dir = "./datasets_gen/data/"
db_stats_save_path = "./db_stats_data/"
workload_name_placehoder = "{workload_name}"
zero_shot_workload_path = f"./zero-shot-data/runs/raw/{workload_name_placehoder}/complex_workload_200k_s1_c8220.json"
dataset_gen_config_path = "./datasets_gen/data_gen_config.json"
imdb_dir = './join-order-benchmark/'

# WORKLOAD NAME
TPCH = "tpch"
TPCDS = "tpcds"
IMDB = "imdb"
TPCH_PLUS = "tpch_plus"
TPCDS_PLUS = "tpcds_plus"
IMDB_PLUS = "imdb_plus"


# INDEX CONFIG TYPE
AUTO_ADMIN_INDCFG = "auto_admin_indcfg"
PERTURBED_INDCFG = "perturbed_indcfg"
RANDOMWALK_INDCFG = "randomwalk_indcfg"

# auto_admin_parameters
auto_admin_parameters = {
    "max_indexes": 5,
    "max_indexes_naive": 1,
    "max_index_width": 2
}

# ablation feature
WO_STAT = "WO_STAT"
WO_ROWS = "WO_ROWS"
WO_PREDICATE = "WO_PREDICATE"
WO_OUTPUT_COL = "WO_OUTPUT_COL"
WO_HIST = "WO_HIST"