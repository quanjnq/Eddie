import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import json
import os
import logging
import sys
import psycopg2
from util.const_util import db_stats_save_path

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG,
                    format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')


def tran_value_type(data_type, vals_str):
    vals = []
    if vals_str is not None:
        strs = vals_str[1:-1].split(",")
        if data_type in {'bigint', 'integer', 'numeric'}:
            vals = [float(v) for v in strs]
        else:
            vals = [v[1:-1] if "\"" in v else v for v in strs]
    return vals


def collect_stats_tbl(tbl):

    rows_sql = f"select reltuples::numeric from pg_class where relname='{tbl}';"
    cursor.execute(rows_sql)
    rows = cursor.fetchone()
    reltuples = rows[0]
    type_sql = f"SELECT column_name, data_type FROM information_schema.columns WHERE table_name = '{tbl}';"
    cursor.execute(type_sql)
    colname_types = cursor.fetchall()
    col2type = dict()
    for col_typ in colname_types:
        col2type[col_typ[0].strip()] = col_typ[1].strip()

    sql = f"select tablename, attname, null_frac, n_distinct, most_common_vals, most_common_freqs, histogram_bounds from pg_stats where schemaname='public' and tablename='{tbl}';"
    cursor.execute(sql)
    rows = cursor.fetchall()

    stats = {}
    for row in rows:
        tbl_att_name = row[0]+"."+row[1]
        null_frac = row[2]
        n_distinct = row[3]
        most_common_vals = tran_value_type(col2type[row[1].strip()], row[4])
        most_common_freqs = [float(v) for v in row[5]] if row[5] is not None else []
        histogram_bounds = tran_value_type(col2type[row[1].strip()], row[6])
        stats[tbl_att_name] = {"null_frac": null_frac, "n_distinct": n_distinct if n_distinct > 0 else abs(n_distinct) * float(reltuples),
            "most_common_vals": most_common_vals, "most_common_freqs": most_common_freqs, "histogram_bounds": histogram_bounds, "rows": float(reltuples), "data_type": col2type[row[1].strip()]}
    return stats


def collect_db_stats():
    tables = []
    sql = "select tablename from pg_tables where schemaname = 'public';"
    cursor.execute(sql)
    rows = cursor.fetchall()
    for row in rows:
        tables.append(row[0])
    merged_stats = {}
    for tbl in tables:
        tbl_stats = collect_stats_tbl(tbl)
        for k in tbl_stats:
            merged_stats[k] = tbl_stats[k]
    return merged_stats


def main_stats(save_path):
    stats = collect_db_stats()
    with open(save_path, "w") as f:
        json.dump(stats, f)


if __name__ == '__main__':
    db_names = ["indexselection_tpcds___10", "indexselection_tpch___10", "imdbload"]
    for database_name in db_names:
        
        conn = psycopg2.connect( database=database_name )
        conn.autocommit = True
        cursor = conn.cursor()

        print(f"collect db stats {database_name} start")
        file_path = db_stats_save_path + f"{database_name}_stats.json"
        
        dst_dir = os.path.dirname(file_path)
        if not os.path.exists(dst_dir):
            os.makedirs(dst_dir)
            
        main_stats(file_path)
        print(f"collect db stats {database_name} finished")
        print(f"Saved db_stats to: {file_path}")

