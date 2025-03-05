# Eddie

This repository provides code for Eddie: Accurate and Robust Index Benefit Estimation Through Hierarchical and Two-dimensional Feature Representation.


##  Environment

You can use conda to create a virtual environment and install requirements as follow:

```
$ conda create --name eddie python=3.9
$ conda activate eddie
$ pip install -r requirements.txt
```
In addition, you need to set up a PostgresSQL server (version 12.13 is preferred) along with the HypoPG extension (version 1.3.1 is preferred).

## Prepare Datasets
Eddie requires specific datasets for training and evaluation. You have two options to prepare the datasets:

### Option 1: Direct Download
You can download the preprocessed datasets directly from this [data repository](https://osf.io/ezx3b/files) and extract them to the `datasets/` directory.

```
$ unzip '*.zip' -d datasets/
```

### Option 2: Reproduce Step-by-Step

#### Step1: Import Databases
1. **For Benchmark Databases (TPC-DS, TPC-H):** 
```
$ python database_scripts/import_database.py
```

2. **For IMDB Database:** Refer to [join-order-benchmark](https://github.com/gregrahn/join-order-benchmark)

3. **For the pre-training augmented training set:** Following the "Reproduce entire DBGen benchmark" section from [zero-shot-cost-estimation](https://github.com/DataManagementLab/zero-shot-cost-estimation), download the datasets, scale and load them into PostgreSQL.
Note: 1. Manually perform `analyze` for each database to update the database statistics. 2. Manually install hypo_pg 1.3.1 for each data, see: https://github.com/HypoPG/hypopg

#### Step2: Collect database statistics
```
$ python database_scripts/collect_db_stats.py
$ python database_scripts/collect_histogram.py
```
Before collecting database statistics, make sure that you can run the analyze command on the target database.

#### Step3: Generate Workloads
For the pre-training augmented training set, we reuse the `complex_workload` from the [data repository](https://osf.io/ga2xj/?view_only=60916f2446e844edb32e3b4d676cb6ee) provided by [zero-shot-cost-estimation](https://github.com/DataManagementLab/zero-shot-cost-estimation), sampling 1000 queries per dataset (e.g., airline, ssb). So please download all files from the zero-shot-data directory in the repository and place them in a local `zero-shot-data/` directory.

And Then execute the following script to generate all workloads used in the experiment, including native workloads, synthetic workloads, and pre-trained workloads:

```
$ python workload_gen/gen_workload.py
```
After execution, the generated workloads will be available for use in subsequent steps.

#### Step4: Generate Datasets
```
$ python datasets_gen/gen_datasets.py
```
Once executed, all datasets used in the experiment will be generated, note that the script may take a long time to execute.

## Train and evaluate Eddie
To train and evaluate Eddie, edit the configuration file located at `exp_configs/exp_config.json` and run the main script.

### Example Configuration (exp_configs/exp_config.json)
The repository includes an example config at `exp_configs/exp_config.json`. Below is a simplified version with comments for reference:

```
{
    "version": "v1", // Config version for tracking changes
    "databases": [
        {
            "db_id": "tpcds_10", // Database identifier: [TPC-DS, SF=10]
            "db_name": "indexselection_tpcds___10", // PostgreSQL database name
            "db_stat_path": "./db_stats_data/indexselection_tpcds___10_stats.json",
            "hist_file_path": "./db_stats_data/indexselection_tpcds___10_hist_file.csv"
        },
        {
            "db_id": "tpcds_5", // [TPC-DS, SF=5]
            "db_name": "indexselection_tpcds___5", 
            "db_stat_path": "./db_stats_data/indexselection_tpcds___5_stats.json",
            "hist_file_path": "./db_stats_data/indexselection_tpcds___5_hist_file.csv"
        }
        // Add more databases as needed (e.g., tpch_10, tpch_5, imdb)
    ],
    "workloads": [
        {
            "workload_name": "tpcds", // Identify different workloads
            "db_id": "tpcds_10", // Corresponding database identifier
            "experiments": [
                {
                    "exp_id": "tpcds__base_wo_init_idx", // Experiment identifier
                    "dataset_path":  "./datasets/tpcds__base_wo_init_idx.pickle",
                    "run_models": ["eddie", "lib", "queryformer", "postgresql"] // Models to run
                },
                {
                    "exp_id": "tpcds__base_w_init_idx",
                    "dataset_path":  "./datasets/tpcds__base_w_init_idx.pickle",
                    "run_models": ["eddie", "lib", "queryformer", "postgresql"]
                },
                {
                    "exp_id": "tpcds__vary_query",
                    "parent_exp_id": "tpcds__base_w_init_idx", // Depends on parent experiment
                    "vary_dataset_path":  "./datasets/tpcds__vary_query.pickle", // Varying dataset for testing
                    "run_models": ["eddie", "lib", "queryformer", "postgresql"]
                },
                {
                    "exp_id": "tpcds__vary_stat",
                    "parent_exp_id": "tpcds__base_w_init_idx",
                    "vary_db_id": "tpcds_5", // Database for varying statistics
                    "vary_dataset_path":  "./datasets/tpcds__vary_stat.pickle",
                    "run_models": ["eddie", "lib", "queryformer", "postgresql"]
                },
                {
                    "exp_id": "tpcds__vary_schema",
                    "parent_exp_id": "tpcds__base_w_init_idx",
                    "vary_schema": true, // Vary schema in experiment
                    "run_models": ["eddie", "lib", "queryformer", "postgresql"]
                },
                {
                    "exp_id": "tpcds__perturb_idx",
                    "dataset_path":  "./datasets/tpcds__perturb_idx.pickle",
                    "run_models": ["eddie", "lib", "queryformer", "postgresql"]
                }
            ]
        }
        // Add more workloads and experiments as needed
    ]
}
```

### Run script
Run the main script to train and evaluate Eddie with the specified configuration:
```
$ python main.py --config exp_configs/exp_config.json
```
After execution, you can view the final metrics in the log file named after the corresponding experiment, located in the `./logs` directory.