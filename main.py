import json
from run_eddie import main as eddie_main
from run_lib import main as lib_main
from run_queryformer import main as qf_main
from run_pg_what_if_est import main as whatif_main
import argparse


entrypoint_map  = {
    "eddie": eddie_main, 
    "lib": lib_main, 
    "queryformer": qf_main, 
    "postgresql": whatif_main
}

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default="./exp_configs/exp_config.json", help='path to experiment config file')
    args = parser.parse_args()

    exp_config_path = args.config
    print("load exp_config from", exp_config_path)
    with open(exp_config_path, "r") as f:
        exp_config = json.load(f)
    version = exp_config["version"]
    
    db_id2info = {}
    for db_cfg in exp_config["databases"]:
        db_id2info[db_cfg["db_id"]] = db_cfg
    
    for workload_info in exp_config["workloads"]:
        base_db_id  = workload_info["db_id"]
        experiments  = workload_info["experiments"]
        workload_name = workload_info["workload_name"]
        
        run_cfg_map = {}
        for exp in experiments:
            for model_name in exp["run_models"]:
                exp_id = exp['exp_id']
                
                run_cfg = {}
                run_cfg["run_id"] = exp_id + '__' + model_name + '_' + version
                run_cfg["model_name"] = model_name
                run_cfg["workload_name"] = workload_name

                if not 'parent_exp_id' in exp:
                    run_cfg["checkpoints_path"] = f"./checkpoints/{run_cfg['run_id']}"
                    run_cfg["dataset_path"] = exp['dataset_path']
                    run_cfg["db_stat_path"] = db_id2info[base_db_id]['db_stat_path']
                    run_cfg["hist_file_path"] = db_id2info[base_db_id]['hist_file_path']
                    run_cfg["db_name"] = db_id2info[base_db_id]['db_name']
                else:
                    run_cfg["is_child_exp"] = True
                    parent_run_cfg = run_cfg_map[exp['parent_exp_id']]
                    # use parent exp checkpoint and dataset
                    run_cfg["checkpoints_path"] = parent_run_cfg['checkpoints_path'] 
                    run_cfg["dataset_path"] = parent_run_cfg['dataset_path']
                    run_cfg["db_stat_path"] = parent_run_cfg['db_stat_path']
                    run_cfg["hist_file_path"] = parent_run_cfg['hist_file_path']
                    run_cfg["db_name"] = parent_run_cfg['db_name']
                    
                    # load new dataset if exists
                    run_cfg["new_dataset_path"] = exp['vary_dataset_path'] if 'vary_dataset_path' in exp else run_cfg["dataset_path"]
                    run_cfg["new_db_stat_path"] = db_id2info[exp['vary_db_id']]['db_stat_path'] if 'vary_db_id' in exp else run_cfg["db_stat_path"]
                    run_cfg["new_hist_file_path"] = db_id2info[exp['vary_db_id']]['hist_file_path'] if 'vary_db_id' in exp else run_cfg["hist_file_path"]
                    run_cfg["new_db_name"] = db_id2info[exp['vary_db_id']]['db_name'] if 'vary_db_id' in exp else run_cfg["db_name"]
                    run_cfg["vary_schema"] = exp['vary_schema'] if 'vary_schema' in exp else False
                
                run_cfg_map[exp_id] = run_cfg
                
                if not exp.get('skip'):
                    entrypoint_map[model_name](run_cfg)
                
# nohup python ./main.py > ./main.log 2>&1 &
