# Eddie

This repository provides code for Eddie: Accurate and Robust Index Benefit Estimation Through Hierarchical and Two-dimensional Feature Representation.


##  Environment

You can use conda to create a virtual environment and install requirements as follow:

```
$ conda create --name eddie python=3.9
$ conda activate eddie
$ pip install -r requirements.txt
```
In addition, you need to set up a PostgresSQL server (version 12.13 is preferred) along with the [HypoPG](https://github.com/HypoPG/hypopg) extension (version 1.3.1 is preferred).

**Cloning the Repository**

This project includes several submodules, which are necessary for running experiments. After cloning the repository, make sure to initialize and update the submodules by running:
```
$ git submodule update --init --recursive
```

## Prepare Datasets
Eddie requires specific datasets for training and evaluation. You have two options to prepare the datasets:

### Option 1: Direct Download
You can download the preprocessed datasets directly from this [data repository](https://osf.io/ezx3b/files) and extract them to the `datasets/` directory in the project’s root directory.

```
$ unzip '*.zip' -d datasets/
```
The directory structure after unzipping will look like this:
```
├── datasets
    ├── tpcds__base_w_init_idx.pickle
    ├── tpcds__base_wo_init_idx.pickle
    └── ...
```

### Option 2: Reproduce Step-by-Step

#### Step1: Import Databases
1. **For Benchmark Databases (TPC-DS, TPC-H):** 
Use the provided script to import TPC-DS and TPC-H databases into PostgreSQL:
```
$ python database_scripts/import_database.py
```

2. **For IMDB Database:** 
Refer to the instructions in the [join-order-benchmark](https://github.com/gregrahn/join-order-benchmark) repository to set up the IMDB database.

3. **For the pre-training augmented training set:** 
Following the "Reproduce entire DBGen benchmark" section from [zero-shot-cost-estimation](https://github.com/DataManagementLab/zero-shot-cost-estimation). Start by downloading all files from the `zero-shot-data` directory in the repository and place them in a local `zero-shot-data/` directory, which will be used in subsequent steps. The directory structure should look like this:
```
├── zero-shot-data
    ├── datasets
        ├── airline
        └── ...
    └── runs
        ├── parsed_plans
            ├── airline
            └── ...
        └── raw
            ├── airline
            └── ...
        └── ...
```
Then, follow the instructions in the referenced section to run scripts for downloading additional data, and scale and load the datasets into PostgreSQL.

#### Step2: Collect database statistics
Before collecting statistics, ensure you can run the `ANALYZE` command on the target database. 

Then execute the following scripts to collect database statistics and histograms:
```
$ python database_scripts/collect_db_stats.py
$ python database_scripts/collect_histogram.py
```

#### Step3: Generate Workloads
Execute the following script to generate all workloads used in the experiment, including native workloads, synthetic workloads, and pre-trained workloads:
```
$ python workload_gen/gen_workload.py
```
After execution, the generated workloads will be available for use in subsequent steps.

#### Step4: Generate Datasets
Generate all datasets required for the experiment with the following script:
```
$ python datasets_gen/gen_datasets.py
```
Note that this script may take a long time to execute. After completion, the resulting datasets will be stored in the datasets/ directory in the project’s root directory, matching the structure from Option 1:
```
├── datasets
    ├── tpcds__base_w_init_idx.pickle
    ├── tpcds__base_wo_init_idx.pickle
    └── ...
```

## Train and evaluate Eddie
Eddie can be trained and evaluated using two methods:

1. **Configuration-based Execution:** Define and modify experiments in `exp_configs/exp_config.json`, then run `main.py`.
2. **Command-line Execution:** Use `run_eddie.py` with specified parameters for flexible experiment execution.

### Method 1: Configuration-based Execution
This method allows you to run experiments using predefined configuration files. Simply specify the configuration file, and the script will automatically load the necessary parameters.

1. **Edit Configuration:**

The repository includes an example configuration at `exp_configs/exp_config.json`. Below is a simplified version with comments for reference:
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

2. **Run the Experiment:**

Run the main script to train and evaluate Eddie with the specified configuration:
```
$ python main.py --config exp_configs/exp_config.json
```
After execution, you can view the final metrics in the log file named after the corresponding experiment, located in the `./logs` directory.

### Method 2: Command-line Execution
For a flexible approach, you can run Eddie by specifying parameters directly via run_eddie.py. Depending on the experiment type, different parameters or configurations may apply. Below are the details for running general experiments, pre-training experiments, and end-to-end experiments.

#### General Experiment
Run a standard experiment with K-fold cross-validation using `run_eddie.py`.

Example Usage:
```sh
$ python run_eddie.py \
    --run_id tpcds__base_w_init_idx__eddie_v1 \
    --model_name eddie \
    --dataset_path ./datasets/tpcds__base_w_init_idx.pickle \
    --db_stat_path ./db_stats_data/indexselection_tpcds___10_stats.json \
    --checkpoints_path ./checkpoints/tpcds__base_w_init_idx__eddie_v1
```

Available Parameters:
- `run_id` (required): Unique identifier for this run. This will be used for logging and checkpoint naming.
- `model_name` (required): Name of the model to be used.
- `dataset_path` (required): Path to the dataset used for K-fold cross-validation.
- `db_stat_path` (required): Path to the database statistics file.
- `checkpoints_path` (required): Directory to save or load model checkpoints. If the specified checkpoint does not exist, the model will be trained during K-fold cross-validation.
- `vary_dataset_path` (optional): Path to a dataset for testing drift scenarios (e.g., changes in queries or data statistics).
- `vary_db_stat_path` (optional): Path to alternative database statistics, used when data statistics change in drift scenarios.
- `vary_schema` (optional): Whether to vary the schema of the dataset, for drift scenarios testing.

After execution, metrics and logs will be stored in the `./logs` directory, with filenames prefixed by the specified run_id. Model checkpoints will be saved in the provided `checkpoints_path`.

#### End-to-End Experiment

**Step 1: Train the Model**
This step is identical to the **General Experiment**, using `run_eddie.py` to train the model for the end-to-end pipeline.

```sh
$ python run_eddie.py \
    --run_id tpcds__end2end_eddie_v1 \
    --model_name eddie \
    --dataset_path ./datasets/tpcds__end2end.pickle \
    --db_stat_path ./db_stats_data/indexselection_tpcds___10_stats.json \
    --checkpoints_path ./checkpoints/tpcds__end2end__eddie_v1
```
The generated checkpoints will be used in the next step.

**Step 2: Run End-to-End Evaluation**
After training, perform end-to-end evaluation with `run_end2end.py`. This script supports evaluating multiple models by specifying them in `--run_models`.

```sh
$ python end2end/run_end2end.py \
    --run_models eddie \
    --db_name indexselection_tpcds___10 \
    --dataset_path ./datasets/tpcds__end2end.pickle \
    --db_stat_path ./db_stats_data/indexselection_tpcds___10_stats.json \
    --checkpoints_path ./checkpoints/tpcds__end2end__{model_name}_v1
```

Available Parameters:

- run_models (required): Comma-separated list of models to evaluate (e.g., "eddie,lib").
- db_name (required): Name of the PostgreSQL database (e.g., "indexselection_tpcds__10").
- dataset_path (required): Path to the dataset for evaluation.
- db_stat_path (required): Path to the database statistics file.
- checkpoints_path (required): Directory template to load model checkpoints; {model_name} is a placeholder replaced by each model name from --run_models (e.g., ./checkpoints/tpcds__end2end__eddie_v1 for "eddie").

After execution, metrics and logs for each model are saved in the `./logs` directory.

#### Pre-training Experiment
**Step 1: Customize Training Parameters**
Before the experiment, you can customize the training parameters in the file `pretrain_finetune/pretrain_finetuneconfig.py`, and the default parameters are the ones used in the paper

**Step 2: Pre-train the Model**
```sh
$ python pretrain_finetune/run_pretrain_eddie.py \
    --run_id pretrain_tpcds \
    --model_name eddie \
    --checkpoints_path ./checkpoints/pretrain_tpcds \
    --workload_name tpcds
```

Available Parameters:
- `run_id` (required): Unique identifier for this run. This will be used for logging and checkpoint naming.
- `model_name` (required): Unique identifier for this run. This will be used for logging and checkpoint naming.
- `checkpoints_path` (required): Unique identifier for this run. This will be used for logging and checkpoint naming.
- `workload_name` (required): Unique identifier for this run. This will be used for logging and checkpoint naming.


**Step 3: Fine-tune the Model**
```sh
$ python pretrain_finetune/run_finetune_eddie.py \
    --run_id tpcds__finetune \
    --model_name eddie \
    --checkpoints_path ./checkpoints/finetune_tpcds \
    --workload_name tpcds \
    --pretrain_model_path ./checkpoints/pretrain_tpcds/tpcds__eddie.pth \
    --finetune_data_rate 0.5
```

Available Parameters:
- `run_id` (required): Unique identifier for this run. This will be used for logging and checkpoint naming.
- `model_name` (required): Unique identifier for this run. This will be used for logging and checkpoint naming.
- `checkpoints_path` (required): Unique identifier for this run. This will be used for logging and checkpoint naming.
- `workload_name` (required): Unique identifier for this run. This will be used for logging and checkpoint naming.
- `pretrain_model_path` (optional): Used to find pre-trained model that have been saved. Parameters used for initializing fine-tuning model
- `finetune_data_rate` (optional): Fine-tune the target data ratio of the model, with a value range of 0-1