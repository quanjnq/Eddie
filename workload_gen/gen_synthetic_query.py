import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import json
from workload_gen.fuzz import Fuzz
from urllib import parse
import pickle
import random
from selection.dbms.postgres_dbms import PostgresDatabaseConnector
from util.workload_util import tran_workload

import logging
import pickle
from util.const_util import *


logging.basicConfig(stream=sys.stdout, level=logging.DEBUG,
                    format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')

class WorkloadArgs:
    def __init__(self, run_params):
        common_prob_conf = {
            "order": 900,
            "limit": 500,
            "group": 900,
            "left": 500,
            "inner": 500,
            "full": 500,
            "scalar": 500,
            "true": 500,
            "func_expr": 50,
            "literal_column": 500,
            "distinct": 100,
            "set": 500,
            "offset": 500,
            "simple": 500,
            "window": 500,
            "extractyear": 50,
            "extractmonth": 50,
            "subquery": 500,
            "nested": 500,
            "where": 100,
            "logical_or":50
        }


        common_query_conf = {"col_backlist":[], "tbl_backlist": [], "min_orderby_cols": 1, "max_orderby_cols": 3, "min_groupby_cols": 1, "max_groupby_cols": 3, "max_selectable_cols": 3, "where_max_predicate_num": 3}
        
        user = "postgres"
        password = parse.quote_plus("M6#P@+dyzTB+tQiV")
        port = 54321
        host = "localhost"
        
        dbname = run_params["dbname"]
        workload_name = run_params["workload_name"]
        
        self.dbname = dbname
        self.connection_string = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{dbname}"
        schema_params = get_params_by_workload_name(workload_name)
        
        self.workload_name = workload_name
        self.join_keys = schema_params["join_keys"]
        self.db_stats = schema_params["db_stats"]
        common_query_conf["db_stats"] = schema_params["db_stats"]
        self.common_prob_conf = common_prob_conf
        self.common_query_conf = common_query_conf
        self.total_query_count = run_params["query_total_count"] if "query_total_count" in run_params else 10
        self.query_max_cost = 1000000
        self.query_min_cost = 0
        
        self.save_base_dir = "./sql_gen/data/workload/"
        # Check if the folder exists
        if not os.path.exists(self.save_base_dir):
            # Create the folder
            os.makedirs(self.save_base_dir)



def gen_query_list(query_prob_conf, common_query_conf, join_keys, db_connector, query_count, args: WorkloadArgs):
    fuzz = Fuzz(query_prob_conf, common_query_conf, join_keys, args.connection_string, db_connector)
    queries = fuzz.gen_orm_queries(query_count, args.query_max_cost, args.query_min_cost)
    return queries


def get_db_conn_with_no_index(dbname):
    db_connector = PostgresDatabaseConnector(dbname, autocommit=True)
    db_connector.drop_indexes()
    db_connector.commit()
    return db_connector


def save_synthetic_querys(queries, filepath_no_suffix, conn):
    workload = tran_workload(queries, conn)
    pickle_path = filepath_no_suffix + ".pickle"
    with open(pickle_path, "wb") as f:
        pickle.dump(workload, f)
        logging.info(f"saved synthetic workload to {pickle_path}")
        
    json_path = filepath_no_suffix + ".json"
    with open(json_path, "w") as f:
        json.dump(queries, f)
        logging.info(f"saved synthetic queries text to {json_path}")
    return pickle_path


def gen_ext_workload(db_connector, args: WorkloadArgs):
    seed = 0
    random.seed(seed)
    workload_name = args.workload_name
    db_name = args.dbname
    total_query_count = args.total_query_count
    save_base_dir = args.save_base_dir
    common_prob_conf = args.common_prob_conf.copy()
    common_query_conf = args.common_query_conf.copy()
    
    logging.info(f"Generating {workload_name} ext workload => seed: {seed}, workload_name: {workload_name},  dbname: {db_name}, prob_conf: {common_prob_conf}")
    sqls = gen_query_list(common_prob_conf, common_query_conf, args.join_keys, db_connector, total_query_count, args)
    
    file_path_no_suffix = save_base_dir + f"{db_name}_syn{len(sqls)}q"
    workload_path = save_synthetic_querys(sqls, file_path_no_suffix, db_connector)
    return workload_path


def get_params_by_workload_name(workload_name):
    workload_name_params = {}
    if workload_name == TPCDS:
        joinkey_tpcds = {'customer_address': {'catalog_returns': [('ca_address_sk', 'cr_returning_addr_sk')], 'catalog_sales': [('ca_address_sk', 'cs_ship_addr_sk')], 'customer': [('ca_address_sk', 'c_current_addr_sk')], 'store_returns': [('ca_address_sk', 'sr_addr_sk')], 'store_sales': [('ca_address_sk', 'ss_addr_sk')], 'web_returns': [('ca_address_sk', 'wr_returning_addr_sk')], 'web_sales': [('ca_address_sk', 'ws_ship_addr_sk')]}, 'customer_demographics': {'catalog_returns': [('cd_demo_sk', 'cr_returning_cdemo_sk')], 'catalog_sales': [('cd_demo_sk', 'cs_ship_cdemo_sk')], 'customer': [('cd_demo_sk', 'c_current_cdemo_sk')], 'store_returns': [('cd_demo_sk', 'sr_cdemo_sk')], 'store_sales': [('cd_demo_sk', 'ss_cdemo_sk')], 'web_returns': [('cd_demo_sk', 'wr_returning_cdemo_sk')], 'web_sales': [('cd_demo_sk', 'ws_ship_cdemo_sk')]}, 'date_dim': {'call_center': [('d_date_sk', 'cc_open_date_sk')], 'catalog_page': [('d_date_sk', 'cp_start_date_sk')], 'catalog_returns': [('d_date_sk', 'cr_returned_date_sk')], 'catalog_sales': [('d_date_sk', 'cs_sold_date_sk')], 'customer': [('d_date_sk', 'c_first_shipto_date_sk')], 'inventory': [('d_date_sk', 'inv_date_sk')], 'promotion': [('d_date_sk', 'p_start_date_sk')], 'store': [('d_date_sk', 's_closed_date_sk')], 'store_returns': [('d_date_sk', 'sr_returned_date_sk')], 'store_sales': [('d_date_sk', 'ss_sold_date_sk')], 'web_page': [('d_date_sk', 'wp_creation_date_sk')], 'web_returns': [('d_date_sk', 'wr_returned_date_sk')], 'web_sales': [('d_date_sk', 'ws_sold_date_sk')], 'web_site': [('d_date_sk', 'web_open_date_sk')]}, 'warehouse': {'catalog_returns': [('w_warehouse_sk', 'cr_warehouse_sk')], 'catalog_sales': [('w_warehouse_sk', 'cs_warehouse_sk')], 'inventory': [('w_warehouse_sk', 'inv_warehouse_sk')], 'web_sales': [('w_warehouse_sk', 'ws_warehouse_sk')]}, 'ship_mode': {'catalog_returns': [('sm_ship_mode_sk', 'cr_ship_mode_sk')], 'catalog_sales': [('sm_ship_mode_sk', 'cs_ship_mode_sk')], 'web_sales': [('sm_ship_mode_sk', 'ws_ship_mode_sk')]}, 'time_dim': {'catalog_returns': [('t_time_sk', 'cr_returned_time_sk')], 'catalog_sales': [('t_time_sk', 'cs_sold_time_sk')], 'store_returns': [('t_time_sk', 'sr_return_time_sk')], 'store_sales': [('t_time_sk', 'ss_sold_time_sk')], 'web_returns': [('t_time_sk', 'wr_returned_time_sk')], 'web_sales': [('t_time_sk', 'ws_sold_time_sk')]}, 'reason': {'catalog_returns': [('r_reason_sk', 'cr_reason_sk')], 'store_returns': [('r_reason_sk', 'sr_reason_sk')], 'web_returns': [('r_reason_sk', 'wr_reason_sk')]}, 'income_band': {'household_demographics': [('ib_income_band_sk', 'hd_income_band_sk')]}, 'item': {'catalog_returns': [('i_item_sk', 'cr_item_sk')], 'catalog_sales': [('i_item_sk', 'cs_item_sk')], 'inventory': [('i_item_sk', 'inv_item_sk')], 'promotion': [('i_item_sk', 'p_item_sk')], 'store_returns': [('i_item_sk', 'sr_item_sk')], 'store_sales': [('i_item_sk', 'ss_item_sk')], 'web_returns': [('i_item_sk', 'wr_item_sk')], 'web_sales': [('i_item_sk', 'ws_item_sk')]}, 'store': {'date_dim': [('s_closed_date_sk', 'd_date_sk')], 'store_returns': [('s_store_sk', 'sr_store_sk')], 'store_sales': [('s_store_sk', 'ss_store_sk')]}, 'call_center': {'date_dim': [('cc_open_date_sk', 'd_date_sk')], 'catalog_returns': [('cc_call_center_sk', 'cr_call_center_sk')], 'catalog_sales': [('cc_call_center_sk', 'cs_call_center_sk')]}, 'customer': {'catalog_returns': [('c_customer_sk', 'cr_returning_customer_sk')], 'catalog_sales': [('c_customer_sk', 'cs_ship_customer_sk')], 'customer_address': [('c_current_addr_sk', 'ca_address_sk')], 'customer_demographics': [('c_current_cdemo_sk', 'cd_demo_sk')], 'household_demographics': [('c_current_hdemo_sk', 'hd_demo_sk')], 'date_dim': [('c_first_shipto_date_sk', 'd_date_sk')], 'store_returns': [('c_customer_sk', 'sr_customer_sk')], 'store_sales': [('c_customer_sk', 'ss_customer_sk')], 'web_returns': [('c_customer_sk', 'wr_returning_customer_sk')], 'web_sales': [('c_customer_sk', 'ws_ship_customer_sk')]}, 'web_site': {'web_sales': [('web_site_sk', 'ws_web_site_sk')], 'date_dim': [('web_open_date_sk', 'd_date_sk')]}, 'store_returns': {'customer_address': [('sr_addr_sk', 'ca_address_sk')], 'customer_demographics': [('sr_cdemo_sk', 'cd_demo_sk')], 'customer': [('sr_customer_sk', 'c_customer_sk')], 'household_demographics': [('sr_hdemo_sk', 'hd_demo_sk')], 'item': [('sr_item_sk', 'i_item_sk')], 'reason': [('sr_reason_sk', 'r_reason_sk')], 'date_dim': [('sr_returned_date_sk', 'd_date_sk')], 'time_dim': [('sr_return_time_sk', 't_time_sk')], 'store': [('sr_store_sk', 's_store_sk')]}, 'household_demographics': {'catalog_returns': [('hd_demo_sk', 'cr_returning_hdemo_sk')], 'catalog_sales': [('hd_demo_sk', 'cs_ship_hdemo_sk')], 'customer': [('hd_demo_sk', 'c_current_hdemo_sk')], 'income_band': [('hd_income_band_sk', 'ib_income_band_sk')], 'store_returns': [('hd_demo_sk', 'sr_hdemo_sk')], 'store_sales': [('hd_demo_sk', 'ss_hdemo_sk')], 'web_returns': [('hd_demo_sk', 'wr_returning_hdemo_sk')], 'web_sales': [('hd_demo_sk', 'ws_ship_hdemo_sk')]}, 'web_page': {'date_dim': [('wp_creation_date_sk', 'd_date_sk')], 'web_returns': [('wp_web_page_sk', 'wr_web_page_sk')], 'web_sales': [('wp_web_page_sk', 'ws_web_page_sk')]}, 'promotion': {'catalog_sales': [('p_promo_sk', 'cs_promo_sk')], 'date_dim': [('p_start_date_sk', 'd_date_sk')], 'item': [('p_item_sk', 'i_item_sk')], 'store_sales': [('p_promo_sk', 'ss_promo_sk')], 'web_sales': [('p_promo_sk', 'ws_promo_sk')]}, 'catalog_page': {'date_dim': [('cp_start_date_sk', 'd_date_sk')], 'catalog_returns': [('cp_catalog_page_sk', 'cr_catalog_page_sk')], 'catalog_sales': [('cp_catalog_page_sk', 'cs_catalog_page_sk')]}, 'inventory': {'date_dim': [('inv_date_sk', 'd_date_sk')], 'item': [('inv_item_sk', 'i_item_sk')], 'warehouse': [('inv_warehouse_sk', 'w_warehouse_sk')]}, 'catalog_returns': {'call_center': [('cr_call_center_sk', 'cc_call_center_sk')], 'catalog_page': [('cr_catalog_page_sk', 'cp_catalog_page_sk')], 'item': [('cr_item_sk', 'i_item_sk')], 'reason': [('cr_reason_sk', 'r_reason_sk')], 'customer_address': [('cr_returning_addr_sk', 'ca_address_sk')], 'customer_demographics': [('cr_returning_cdemo_sk', 'cd_demo_sk')], 'customer': [('cr_returning_customer_sk', 'c_customer_sk')], 'household_demographics': [('cr_returning_hdemo_sk', 'hd_demo_sk')], 'date_dim': [('cr_returned_date_sk', 'd_date_sk')], 'time_dim': [('cr_returned_time_sk', 't_time_sk')], 'ship_mode': [('cr_ship_mode_sk', 'sm_ship_mode_sk')], 'warehouse': [('cr_warehouse_sk', 'w_warehouse_sk')]}, 'web_returns': {'item': [('wr_item_sk', 'i_item_sk')], 'reason': [('wr_reason_sk', 'r_reason_sk')], 'customer_address': [('wr_returning_addr_sk', 'ca_address_sk')], 'customer_demographics': [('wr_returning_cdemo_sk', 'cd_demo_sk')], 'customer': [('wr_returning_customer_sk', 'c_customer_sk')], 'household_demographics': [('wr_returning_hdemo_sk', 'hd_demo_sk')], 'date_dim': [('wr_returned_date_sk', 'd_date_sk')], 'time_dim': [('wr_returned_time_sk', 't_time_sk')], 'web_page': [('wr_web_page_sk', 'wp_web_page_sk')]}, 'web_sales': {'customer_address': [('ws_ship_addr_sk', 'ca_address_sk')], 'customer_demographics': [('ws_ship_cdemo_sk', 'cd_demo_sk')], 'customer': [('ws_ship_customer_sk', 'c_customer_sk')], 'household_demographics': [('ws_ship_hdemo_sk', 'hd_demo_sk')], 'item': [('ws_item_sk', 'i_item_sk')], 'promotion': [('ws_promo_sk', 'p_promo_sk')], 'date_dim': [('ws_sold_date_sk', 'd_date_sk')], 'ship_mode': [('ws_ship_mode_sk', 'sm_ship_mode_sk')], 'time_dim': [('ws_sold_time_sk', 't_time_sk')], 'warehouse': [('ws_warehouse_sk', 'w_warehouse_sk')], 'web_page': [('ws_web_page_sk', 'wp_web_page_sk')], 'web_site': [('ws_web_site_sk', 'web_site_sk')]}, 'catalog_sales': {'customer_address': [('cs_ship_addr_sk', 'ca_address_sk')], 'customer_demographics': [('cs_ship_cdemo_sk', 'cd_demo_sk')], 'customer': [('cs_ship_customer_sk', 'c_customer_sk')], 'household_demographics': [('cs_ship_hdemo_sk', 'hd_demo_sk')], 'call_center': [('cs_call_center_sk', 'cc_call_center_sk')], 'catalog_page': [('cs_catalog_page_sk', 'cp_catalog_page_sk')], 'item': [('cs_item_sk', 'i_item_sk')], 'promotion': [('cs_promo_sk', 'p_promo_sk')], 'date_dim': [('cs_sold_date_sk', 'd_date_sk')], 'ship_mode': [('cs_ship_mode_sk', 'sm_ship_mode_sk')], 'time_dim': [('cs_sold_time_sk', 't_time_sk')], 'warehouse': [('cs_warehouse_sk', 'w_warehouse_sk')]}, 'store_sales': {'customer_address': [('ss_addr_sk', 'ca_address_sk')], 'customer_demographics': [('ss_cdemo_sk', 'cd_demo_sk')], 'customer': [('ss_customer_sk', 'c_customer_sk')], 'household_demographics': [('ss_hdemo_sk', 'hd_demo_sk')], 'item': [('ss_item_sk', 'i_item_sk')], 'promotion': [('ss_promo_sk', 'p_promo_sk')], 'date_dim': [('ss_sold_date_sk', 'd_date_sk')], 'time_dim': [('ss_sold_time_sk', 't_time_sk')], 'store': [('ss_store_sk', 's_store_sk')]}}
        
        tpcds_db_stats_path = "./db_stats_data/indexselection_tpcds___10_stats.json"
        with open(tpcds_db_stats_path, "r") as f:
            db_stats = json.load(f)
            
        workload_name_params["join_keys"] = joinkey_tpcds
        workload_name_params["db_stats"] = db_stats
        
    elif workload_name == TPCH:
        joinkey_tpch = {'part': {'partsupp': [('p_partkey', 'ps_partkey')]}, 'supplier': {'partsupp': [('s_suppkey', 'ps_suppkey')], 'customer': [('s_nationkey', 'c_nationkey')]}, 'partsupp': {'lineitem': [('ps_partkey', 'l_partkey'), ('ps_suppkey', 'l_suppkey')], 'supplier': [('ps_suppkey', 's_suppkey')], 'part': [('ps_partkey', 'p_partkey')]}, 'customer': {'orders': [('c_custkey', 'o_custkey')], 'supplier': [('c_nationkey', 's_nationkey')]}, 'lineitem': {'partsupp': [('l_partkey', 'ps_partkey'), ('l_suppkey', 'ps_suppkey')], 'orders': [('l_orderkey', 'o_orderkey')]}, 'orders': {'lineitem': [('o_orderkey', 'l_orderkey')], 'customer': [('o_custkey', 'c_custkey')]}}
        
        tpcds_db_stats_path = "./db_stats_data/indexselection_tpch___10_stats.json"
        with open(tpcds_db_stats_path, "r") as f:
            db_stats = json.load(f)
            
        workload_name_params["join_keys"] = joinkey_tpch
        workload_name_params["db_stats"] = db_stats
    elif workload_name == IMDB:
        joinkey_imdb = {"aka_title": {"title": [('movie_id', 'id')] }, 'char_name' : {'cast_info': [('id', 'person_role_id')] }, 'role_type': {'cast_info': [('id', 'role_id')]}, 'comp_cast_type': {'complete_cast': [('id', 'subject_id'), ('id', 'status_id')]},
                        "movie_link": {"title": [('id', 'id'), ('movie_id', 'id')] , 'link_type': [('link_type_id', 'id')]}, 'link_type': {'movie_link': [('id', 'link_type_id')]},  "cast_info": {"title": [("movie_id", "id")] , 'aka_name':[("person_id","person_id")], 'char_name':[("person_role_id", "id")], "role_type":[("role_id","id")]},
                        "complete_cast": {"comp_cast_type":[('subject_id', 'id'), ('status_id', 'id')], "title": [("movie_id", "id")] }, "title": {"complete_cast": [("id", "movie_id")], "aka_title": [("id", "movie_id")], "movie_link":[('id', 'id'), ('id', 'movie_id')], "cast_info": [("id", "movie_id")], "movie_companies": [("id", "movie_id")], "movie_keyword": [("id", "movie_id")], "movie_info_idx": [("id", "movie_id")], "movie_info": [("id", "movie_id")], "kind_type": [("id", "id")] } ,
                        "aka_name":{'cast_info':[("person_id","person_id")], "name": [("person_id","id")]}, "movie_companies": {"company_name": [("company_id", "id")], "title": [("movie_id", "id")] }, "kind_type": {"title": [('id', 'id')]}, "name": {"aka_name":[("id", "person_id")], "person_info":[("id", "person_id")]}, "movie_keyword": {"keyword":[("movie_id", "id")]},
                        "info_type":{"movie_info":[("id","info_type_id")], "movie_info_idx":[("id","info_type_id")]}}
        db_stats_path = "./db_stats_data/imdbload_stats.json"
        with open(db_stats_path, "r") as f:
            db_stats = json.load(f)
            
        workload_name_params["join_keys"] = joinkey_imdb
        workload_name_params["db_stats"] = db_stats
    else:
        raise Exception(f"unknow workload_name type {workload_name}")
    return workload_name_params

def main(run_params):
    logging.info(f"Use run_params: {run_params}")
    w_args = WorkloadArgs(run_params)
    db_connector = get_db_conn_with_no_index(w_args.dbname)
    
    workload_file_path = gen_ext_workload(db_connector, w_args)
    
    try:
        db_connector.close()
    except Exception as e:
        logging.info(f"db_connector close exception {e}")
    return workload_file_path
    

if __name__ == "__main__":
    gen_synthetic_query_params = {"dbname": "indexselection_tpcds___10", "workload_name": TPCDS, "query_total_count": 5}
    main(gen_synthetic_query_params)

    

# nohup python ./synthetic_query/run_synthetic_query.py > ./synthetic_query/run_synthetic_query.log 2>&1 &
