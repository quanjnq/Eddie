# From Querformer
import pandas as pd
import json
import numpy as np
import psycopg2
import time
from util.const_util import db_stats_save_path


def to_vals(data_list):
    for dat in data_list:
        val = dat[0]
        if val is not None: break
    try:
        float(val)
        return np.array(data_list, dtype=float).squeeze()
    except:
        res = []
        for dat in data_list:
            try:
                mi = dat[0].timestamp()
            except:
                mi = 0
            res.append(mi)
        return np.array(res)
    
    
def main(db_name, db_stat_path):
    schema = {}
    col2id = {}
    with open(db_stat_path, "r")  as f:
        db_stat = json.load(f)

    for tbl_col in db_stat:
        col2id[tbl_col] = len(col2id) + 1
        tbl, col = tbl_col.split(".")
        if tbl not in schema:
            schema[tbl] = []
        schema[tbl].append(col)
    st = time.time()
    conm = psycopg2.connect(database=db_name)
    conm.set_session(autocommit=True)
    cur = conm.cursor()
    
    hist_file = pd.DataFrame(columns=['table','column','bins','table_column'])
    min_max_vals = {}
    max_min_vals = {}
    for table,columns in schema.items():
        for column in columns:
            cmd = 'select {} from {}'.format(column, table)
            print(cmd)
            cur.execute(cmd)
            col = cur.fetchall()
            col_array = to_vals(col)
            hists = np.nanpercentile(col_array, range(0,101,2), axis=0)
            res_dict = {
                'table':table,
                'column':column,
                'table_column': '.'.join((table, column)),
                'bins':hists
            }
            max_v, min_v =  np.max(hists), np.min(hists)
            min_max_vals[f"{table}.{column}"] = [min_v, max_v]
            max_min_vals[f"{table}.{column}"] = [max_v, min_v]
            hist_file = hist_file._append(res_dict,ignore_index=True)
    fp = db_stats_save_path + f"{db_name}_hist_file.csv"
    hist_file.to_csv()
    print(min_max_vals)
    print("Saved hist_file: ", fp)
    print(f"time cost: {time.time() - st} s")

if __name__ == '__main__':
    
    params = [
        # (db_name, db_stats_data)
        ("indexselection_tpcds___10", './db_stats_data/indexselection_tpcds___10_stats.json'),
        ("indexselection_tpch___10", './db_stats_data/indexselection_tpch___10_stats.json'),
        ("imdbload", './db_stats_data/imdbload_stats.json'),
        ]
    for param in params:
        main(param[0], param[1])
