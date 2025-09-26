import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pickle
import logging
import sys
from selection.dbms.postgres_dbms import PostgresDatabaseConnector
import json
import hashlib
from util.const_util import AUTO_ADMIN_INDCFG, PERTURBED_INDCFG, RANDOMWALK_INDCFG, data_gen_work_dir

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')


def are_dicts_equal(dict1, dict2):
    hash1 = hashlib.md5(json.dumps(dict1).encode()).hexdigest()
    hash2 = hashlib.md5(json.dumps(dict2).encode()).hexdigest()
    return hash1 == hash2


def get_avg_runtimes(query, timeout=180*1000, db_connector=None):
    each_times = []
    times = 1
    for k in range(times):
        actual_runtimes, plan = db_connector.exec_query(query, timeout=timeout)
        if actual_runtimes is None and k > 0:
            return None, None
        each_times.append(actual_runtimes)

    if times > 1:
        avg_times = sum(each_times[1:]) / len(each_times[1:])
    else:
        avg_times = each_times[0]
    return avg_times, plan


def clear_all_indexes(db_connector):
    stmt = "select indexname from pg_indexes where schemaname='public'"
    indexes = db_connector.exec_fetch(stmt, one=False)
    for index in indexes:
        index_name = index[0]
        drop_stmt = 'drop index "{}"'.format(index_name)
        db_connector.exec_only(drop_stmt)
        db_connector.commit()


def get_query_index_pairs(path, ind_tpy):
    with open(path, 'rb') as f:
        qs_indcfgs = pickle.load(f)
    samples = []
    for q_indcfgs in qs_indcfgs:
        if ind_tpy == AUTO_ADMIN_INDCFG:
            samples.append([q_indcfgs["query"], q_indcfgs[AUTO_ADMIN_INDCFG]])
        elif ind_tpy == PERTURBED_INDCFG:
            samples.append([q_indcfgs["query"], q_indcfgs[PERTURBED_INDCFG]])
        elif ind_tpy == RANDOMWALK_INDCFG:
            samples.append([q_indcfgs["query"], q_indcfgs[RANDOMWALK_INDCFG]])
        else:
            raise Exception(f"unknow index type: {ind_tpy}")
    return samples


def create_hypo_indexs(db_connector, indexs):
    for index in indexs:
        if type(index) is str:
            tab_cols = index.split("#")
            statement = f"create index on {tab_cols[0]} ({tab_cols[1]})"
        else:
            statement = f"create index {index.index_idx()} on {index.table().name} ({index.joined_column_names()})"
        statement = f"select * from hypopg_create_index('{statement}')"
        db_connector.exec_only(statement)
        db_connector.commit()

def reset_hypo_indexes(db_connector):
    stmt = "SELECT * FROM hypopg_reset();"
    db_connector.exec_only(stmt)
    db_connector.commit()


def run_gen_sample_label(query_index_pairs, breakpoint_path, start_qi=0, db_connector=None):
    expected_sum_sample_num = sum([len(pair[1]) for pair in query_index_pairs])
    logging.info(f'Expected sample size: {expected_sum_sample_num}')

    clear_all_indexes(db_connector)

    sample_num = 0
    all_samples = []
    timeout_query_list = []
    q_cnt = len(query_index_pairs)
    logging.info(f'Start sample number: {start_qi}')
    save_threshold = 0
    for qi, (query, index_cfgs) in enumerate(query_index_pairs):
        try:
            if qi < start_qi:
                continue
            proced_query_text = query.text.replace("\n"," ")
            print()
            logging.info(f'{query} index_cfgs_num={len(index_cfgs)} {proced_query_text}')
            if len(index_cfgs) == 0:
                continue
            orig_logic_cost, orig_logic_plan = db_connector.exec_explain_query(query)
            orig_runtimes, orig_plan = get_avg_runtimes(query, db_connector=db_connector)
            if orig_runtimes is None:
                timeout_query_list.append(query)
                logging.info(f"orgi query timeout {query.nr} {query.text}")
                continue

            # Traverse through each index configuration
            for index_cfg in index_cfgs:
                # Create an index configuration
                plan_changed = False
                create_hypo_indexs(db_connector, index_cfg)
                # The execution of the index configuration
                logic_cost, logic_plan = db_connector.exec_explain_query(query)
                reset_hypo_indexes(db_connector)
                if are_dicts_equal(logic_plan, orig_logic_plan):
                    # The query plan does not change, using the original time and plan
                    with_index_runtimes, with_index_plan = orig_runtimes, orig_plan
                else:
                    for index in index_cfg:
                        sp = index.split("#")
                        tbl, cols = sp[0], sp[1]
                        cols = [f'"{col}"' for col in cols.split(",")]
                        joined_cols_str = ",".join(cols)
                        statement = f"create index on "+f'"{tbl}"'+f" ({joined_cols_str})"
                        db_connector.exec_only(statement)
                        db_connector.commit()
                    # If the query plan changes, the execution obtains the time and plan after the change
                    timeout = max(int(orig_runtimes)*2, 10*1000)
                    with_index_runtimes, with_index_plan = get_avg_runtimes(query, timeout=timeout, db_connector=db_connector)
                    if with_index_runtimes is None:
                        with_index_runtimes = timeout
                    plan_changed = True

                all_samples.append((query, (tuple(), index_cfg), (orig_runtimes, orig_plan), (with_index_runtimes, with_index_plan)))
                sample_num += 1
                logging.info(f"Query{qi}/{q_cnt} Case{sample_num}/{expected_sum_sample_num} {(query, plan_changed, orig_runtimes, with_index_runtimes)} {index_cfg}")
                clear_all_indexes(db_connector)
            if sample_num > save_threshold:
                save_threshold += 100
                with open(breakpoint_path, 'wb') as f:
                    pickle.dump(all_samples, f)
                    logging.info(f"breakpoint qi={qi} save to {breakpoint_path}")
        except Exception as e:
            logging.info(f"except: {e}")
            clear_all_indexes(db_connector)
    return all_samples, timeout_query_list


def main(run_params):

    database_name = run_params["db_name"]
    path = run_params["query_indcfgs_path"]
    common_name = path.split("/")[-1].split(".")[0]
    
    if "conn_cfg" in run_params:
        conn_cfg = run_params["conn_cfg"]
        db_connector = PostgresDatabaseConnector(database_name, host=conn_cfg["host"],  port=conn_cfg["port"],  user=conn_cfg["user"], password=conn_cfg["password"], autocommit=True)
    else:
        db_connector = PostgresDatabaseConnector(database_name, autocommit=True)

    saved_sample_path_dict = {}
    save_dir_path = data_gen_work_dir + "samples/"
    if not os.path.exists(save_dir_path):
        os.makedirs(save_dir_path)

    for (ind_tpy, breakpoint_num) in run_params["indextype_and_breakpoint"]:
        q_indcfgs_pairs = get_query_index_pairs(path, ind_tpy)
        logging.info(f"{ind_tpy} load pickle from {path}")
        logging.info(f'{ind_tpy} query cnt: {len(q_indcfgs_pairs)}')

        breakpoint_save_path = save_dir_path + f'{common_name}_{ind_tpy}_breakpoint{breakpoint_num}.pickle'

        final_all_samples, final_timeout_query_list = run_gen_sample_label(q_indcfgs_pairs, breakpoint_save_path, start_qi=breakpoint_num, db_connector=db_connector)

        if len(final_timeout_query_list) > 0:
            try:
                logging.info(f"timeout query list: {[q.nr for q in final_timeout_query_list]}")
                save_timieout_path = save_dir_path + f"{common_name}_{ind_tpy}_sqi{breakpoint_num}_timeoutqs_{len(final_timeout_query_list)}.pickle"
                with open(save_timieout_path, 'wb') as f:
                    pickle.dump(final_timeout_query_list, f)
                    logging.info(f"succ save  {ind_tpy} final_timeout_query_list to {save_timieout_path}")
            except:
                logging.info("except save final_timeout_query_list")

        save_path = save_dir_path + f'{common_name}_{ind_tpy}{len(final_all_samples)}_sqi{breakpoint_num}.pickle'
        with open(save_path, 'wb') as f:
            pickle.dump(final_all_samples, f)
            logging.info(f"Succ Saved {ind_tpy} to {save_path}")
            saved_sample_path_dict[ind_tpy] = save_path
            print()
        # Delete the breakpoint file
        if os.path.exists(breakpoint_save_path):
            os.remove(breakpoint_save_path)
            logging.info(f"Deleted breakpoint file {breakpoint_save_path}")

    try:
        db_connector.close()
    except Exception as e:
        logging.info(f"db_connector close Exception {e}")
    return saved_sample_path_dict


if __name__ == '__main__':
    run_params = {"db_name": "indexselection_tpcds___10", "query_indcfgs_path": "./sql_gen/data/candidate_indexes/indexselection_tpcds___10_10q_autoadmin_10_extend_3.pickle", "indextype_and_breakpoint": [(AUTO_ADMIN_INDCFG, 0), (PERTURBED_INDCFG, 0)]}
    main(run_params)

