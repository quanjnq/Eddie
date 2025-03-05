python ./database_scripts/collect_db_stats.py
# python ./database_scripts/collect_histogram.py
python ./workload_gen/gen_workload.py
python ./datasets_gen/gen_datasets.py

# nohup ./database_scripts/data_pipleline.sh > ./database_scripts/main.log 2>&1 &