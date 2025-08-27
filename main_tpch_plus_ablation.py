import json
from run_eddie import main as eddie_main
from run_lib import main as lib_main
from run_queryformer import main as qf_main
from run_pg_what_if_est import main as whatif_main
from end2end.run_end2end import main as end2end_main
from pretrain_finetune.run_pretrain_eddie import main as pretrain_eddie_main
from pretrain_finetune.run_finetune_eddie import main as finetune_eddie_main

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
        ablation_feature = workload_info["ablation_feature"] if "ablation_feature" in workload_info else None
        
        run_cfg_map = {}
        for exp in experiments:
            if exp.get('skip'):
                continue
            
            if "pretrain" not in exp['exp_id'] and "finetune" not in exp['exp_id']: # Pre-training fine-tuning does not follow the main experimental logic
                for model_name in exp["run_models"]:
                    exp_id = exp['exp_id']
                    run_cfg = {}
                    ablation_feature_ = ablation_feature + "_" if ablation_feature else ""
                    run_cfg["run_id"] = exp_id + '__' + model_name + '_' + ablation_feature_ + version
                    run_cfg["model_name"] = model_name
                    run_cfg["workload_name"] = workload_name
                    run_cfg["clip_label"] = "True"
                    run_cfg["ablation_feature"] = ablation_feature

                    if not 'parent_exp_id' in exp:
                        run_cfg["checkpoints_path"] = f"./checkpoints/{run_cfg['run_id']}"
                        run_cfg["dataset_path"] = exp['dataset_path']
                        run_cfg["db_stat_path"] = db_id2info[base_db_id]['db_stat_path']
                        run_cfg["hist_file_path"] = db_id2info[base_db_id]['hist_file_path']
                        run_cfg["db_name"] = db_id2info[base_db_id]['db_name']
                    else:
                        parent_run_id = exp['parent_exp_id'] + '__' + model_name + '_' + version
                        parent_run_cfg = run_cfg_map[parent_run_id]
                        # use parent exp checkpoint and dataset
                        run_cfg["checkpoints_path"] = parent_run_cfg['checkpoints_path']
                        run_cfg["dataset_path"] = parent_run_cfg['dataset_path']
                        run_cfg["db_stat_path"] = parent_run_cfg['db_stat_path']
                        run_cfg["hist_file_path"] = parent_run_cfg['hist_file_path']
                        run_cfg["db_name"] = parent_run_cfg['db_name']
                        
                        # load new dataset if exists
                        run_cfg["vary_dataset_path"] = exp['vary_dataset_path'] if 'vary_dataset_path' in exp else run_cfg["dataset_path"]
                        run_cfg["vary_db_stat_path"] = db_id2info[exp['vary_db_id']]['db_stat_path'] if 'vary_db_id' in exp else run_cfg["db_stat_path"]
                        run_cfg["vary_hist_file_path"] = db_id2info[exp['vary_db_id']]['hist_file_path'] if 'vary_db_id' in exp else run_cfg["hist_file_path"]
                        run_cfg["vary_db_name"] = db_id2info[exp['vary_db_id']]['db_name'] if 'vary_db_id' in exp else run_cfg["db_name"]
                        run_cfg["vary_schema"] = exp['vary_schema'] if 'vary_schema' in exp else False
                    
                    run_cfg_map[run_cfg["run_id"]] = run_cfg
                    
                    if not exp.get('skip'):
                        entrypoint_map[model_name](run_cfg)
            
            run_cfg = {}
            run_cfg.update(exp)
            exp_id = exp['exp_id']
            if "end2end" in exp_id:
                run_cfg["db_name"] = db_id2info[base_db_id]['db_name']
                run_cfg["db_stat_path"] = db_id2info[base_db_id]['db_stat_path']
                run_cfg["version"] = version
                end2end_main(run_cfg)
            elif "pretrain" in exp_id:
                run_cfg["run_id"] = run_cfg["exp_id"] + '__' + run_cfg["model_name"] + '_' + version
                run_cfg["workload_name"] = workload_name
                run_cfg["checkpoints_path"] = f"./checkpoints/{run_cfg['run_id']}"
                pretrain_eddie_main(run_cfg)
            elif "finetune" in exp_id:
                run_cfg["run_id"] = run_cfg["exp_id"] + '__' + run_cfg["model_name"] + '_' + version
                run_cfg["workload_name"] = workload_name
                run_cfg["checkpoints_path"] = f"./checkpoints/{run_cfg['run_id']}"
                run_cfg["parent_run_id"] = exp['parent_exp_id'] + '__' + run_cfg["model_name"] + '_' + version
                finetune_eddie_main(run_cfg)

# nohup python ./main_tpch_plus_ablation.py --config exp_configs/exp_config_tpch_plus_ablation.json > ./main_tpch_plus_ablation.log 2>&1 &