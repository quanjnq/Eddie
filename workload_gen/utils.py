import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from selection.algorithms.auto_admin_algorithm import AutoAdminAlgorithm
from util.workload_util import tran_workload
from util.const_util import auto_admin_parameters



def is_valid_autoadmin_query(db_connector, sql):
    workload = tran_workload([sql], db_connector)
    algorithm = AutoAdminAlgorithm(db_connector, auto_admin_parameters)
    indexes = algorithm.calculate_best_indexes(workload)
    index_conf_old = []
    for ind in indexes:
        index_str = f"{ind.table()}#{ind.joined_column_names()}"
        index_conf_old.append(index_str)
    if cmp_plan_with_conn(workload.queries[0], [index_conf_old], db_connector) == 0:
        return False
    return True


def cmp_plan_with_conn(query, indcfgs, db_connector):
    logic_cost, logic_plan = db_connector.exec_explain_query(query)
    cnt = 0
    for indcfg in indcfgs:
        clear_all_hypo_indexes(db_connector)
        create_hypo_indexes(query, indcfg, db_connector)
        logic_cost_ind, logic_plan_ind = db_connector.exec_explain_query(query)
        if str(logic_plan) != str(logic_plan_ind):
            cnt += 1
    clear_all_hypo_indexes(db_connector)
    return cnt


def create_hypo_indexes(query, indexs, db_connector=None):
    for index in indexs:
        if type(index) is str:
            tab_cols = index.split("#")
            statement = f"create index on {tab_cols[0]} ({tab_cols[1]})"
        else:
            statement = f"create index {index.index_idx()} on {index.table().name} ({index.joined_column_names()})"
        statement = f"select * from hypopg_create_index('{statement}')"
        db_connector.exec_only(statement)
        db_connector.commit()


def clear_all_hypo_indexes(db_connector):
    stmt = "SELECT * FROM hypopg_reset();"
    db_connector.exec_only(stmt)
