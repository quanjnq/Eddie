
import pickle
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from selection.dbms.postgres_dbms import PostgresDatabaseConnector
import pickle
import random
from  util.workload_util import tran_workload
import re
pattern_production_year = r"production_year\s*([!=<>]=?|<|>)\s*(\d+)"
pattern_production_year_btw = r"production_year\s*BETWEEN\s*(\d*)"
pattern_country_code =r"cn\.country_code\s*=\s*'(.+)'"
pattern_info_type = "it\d?.info\s*=\s*'(.+)'"
pattern_keyword = r"k.keyword\s*=\s*'(.+)'"


def get_val(tbl_col, db_stat):
    col_stat = db_stat[tbl_col]
    most_val_list = col_stat["most_common_vals"] if len(col_stat["most_common_vals"]) > 0 else [0]
    comp_num = random.choice(col_stat["histogram_bounds"]) if len(col_stat["histogram_bounds"]) > 0 else most_val_list
    return comp_num


def paramize(sql_text, db_stat):
    match0 = re.search(pattern_production_year, sql_text)
    match1 = re.search(pattern_production_year_btw, sql_text)
    match2 = re.search(pattern_country_code, sql_text)
    match3 = re.search(pattern_info_type, sql_text)
    match4 = re.search(pattern_keyword, sql_text)
    if match0:
        orig_str = match0.group(0)
        old_val = match0.group(2)
        new_val = int(old_val) + random.choice([-2,-1,0,1,2])
        new_str = orig_str.replace(old_val, str(new_val))
        return sql_text.replace(orig_str, new_str)
    if match1:
        orig_str = match1.group(0)
        old_val = match1.group(1)
        new_val = int(old_val) + random.choice([-2,-1,0,1,2])
        new_str = orig_str.replace(old_val, str(new_val))
        return sql_text.replace(orig_str, new_str)
    if match2:
        key = "company_name.country_code"
        new_val = get_val(key, db_stat)
        old_str = match2.group(0)
        old_val = match2.group(1)
        new_str = old_str.replace(old_val, str(new_val))
        return sql_text.replace(old_str, new_str)
    if match3:
        key = "info_type.info"
        new_val = get_val(key, db_stat)
        old_str = match3.group(0)
        old_val = match3.group(1)
        new_str = old_str.replace(old_val, str(new_val))
        return sql_text.replace(old_str, new_str)
    if match4:
        key = "keyword.keyword"
        new_val = get_val(key, db_stat)
        old_str = match4.group(0)
        old_val = match4.group(1)
        new_str = old_str.replace(old_val, str(new_val))
        return sql_text.replace(old_str, new_str)




def paramize_imdb(imdb_nr_sqls_tuples, database_name, db_stat, times= 3):
    random.seed(0)
    db_connector = PostgresDatabaseConnector(database_name, autocommit=True)
    qs = []

    for nr, sql in imdb_nr_sqls_tuples:
        qs.append((nr, sql)) # join self.
        statement = f"explain (format json) {sql}"
        result = db_connector.exec_fetch(statement, one=True)

        old_cost = result[0][0]['Plan']["Total Cost"]

        for i in range(times-1): # The remaining queries are parameterized
            while True:
                newsql = paramize(sql, db_stat)
                statement = f"explain (format json) {newsql}"
                try:
                    result = db_connector.exec_fetch(statement, one=True)
                    new_cost = result[0][0]['Plan']["Total Cost"]
                    if int(new_cost) > int(old_cost) * 1.2: # Prevent the cost from being too high and the cost from being too high
                        continue
                    qs.append((nr, newsql))
                    break
                except:
                    pass

    workload_imdb = tran_workload(qs, db_connector)
    db_connector.close()
    return workload_imdb
