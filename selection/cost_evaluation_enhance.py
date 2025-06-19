import logging

from selection.what_if_index_creation import WhatIfIndexCreation
import hashlib
import json
import pickle
from pathlib import Path
import os


def get_avg_runtimes(query, timeout=120*1000, db_connector=None):
    each_times = []    
    times = 4
    for k in range(times): 
        actual_runtimes, plan = db_connector.exec_query(query, timeout=timeout)
        if actual_runtimes is None and k > 0:
            return timeout, None
        each_times.append(actual_runtimes)

    if times > 1:
        avg_times = sum(each_times[1:]) / len(each_times[1:])
    else:
        avg_times = each_times[0]
    return avg_times, plan


def are_dicts_equal(dict1, dict2):
    hash1 = hashlib.md5(json.dumps(dict1).encode()).hexdigest()
    hash2 = hashlib.md5(json.dumps(dict2).encode()).hexdigest()
    return hash1 == hash2

class CostEvaluation:
    def __init__(self, db_connector, cost_estimation="whatif", sql2plan=None, workload=None, group_num=0, w_path=None):
        self.cur_query = None
        fn = w_path.split("/")[-1].split(".")[0]
        self.db_connector = db_connector
        self.cost_estimation = cost_estimation
        self.what_if = WhatIfIndexCreation(db_connector)
        self.current_indexes = set()
        self.cost_requests = 0
        self.cache_hits = 0
        self.cache = {}
        self.completed = False
        # It is not necessary to drop hypothetical indexes during __init__().
        # These are only created per connection. Hence, non should be present.

        self.relevant_indexes_cache = {}
        self.invoke_cnt = 0
        
        self.sql2plan_tree = sql2plan if sql2plan else {}
        self.query2logic_plan = {}
        for query in workload.queries:
            orig_logic_cost, orig_logic_plan = db_connector.exec_explain_query(query)
            self.query2logic_plan[query] = orig_logic_plan
        self.all_samples = []
        dir = "./enhanced_data"
        self.breakpoint_path = f"{dir}/{fn}_g{group_num}_bp.pickle"
        Path(dir).mkdir(parents=True, exist_ok=True)

    def estimate_size(self, index):
        # TODO: Refactor: It is currently too complicated to compute
        # We must search in current indexes to get an index object with .hypopg_oid
        result = None
        for i in self.current_indexes:
            if index == i:
                result = i
                break
        if result:
            # Index does currently exist and size can be queried
            if not index.estimated_size:
                index.estimated_size = self.what_if.estimate_index_size(result.hypopg_oid)
        else:
            self._simulate_or_create_index(index, store_size=True)

    def which_indexes_utilized_and_cost(self, query, indexes):
        self._prepare_cost_calculation(indexes, store_size=True)

        plan = self.db_connector.get_plan(query)
        cost = plan["Total Cost"]
        plan_str = str(plan)

        recommended_indexes = set()

        # We are iterating over the CostEvalution's indexes and not over `indexes`
        # because it is not guaranteed that hypopg_name is set for all items in
        # `indexes`. This is caused by _prepare_cost_calculation that only creates
        # indexes which are not yet existing. If there is no hypothetical index
        # created for an index object, there is no hypopg_name assigned to it. However,
        # all items in current_indexes must also have an equivalent in `indexes`.
        for index in self.current_indexes:
            assert (
                index in indexes
            ), "Something went wrong with _prepare_cost_calculation."

            if index.hypopg_name not in plan_str:
                continue
            recommended_indexes.add(index)

        return recommended_indexes, cost

    def calculate_cost(self, workload, indexes, store_size=False):
        assert (
            self.completed is False
        ), "Cost Evaluation is completed and cannot be reused."
        self._prepare_cost_calculation(indexes, store_size=store_size)
        total_cost = 0

        # TODO: Make query cost higher for queries which are running often
        for query in workload.queries:
            self.cost_requests += 1
            total_cost += self._request_cache(query, indexes)
        return total_cost

    # Creates the current index combination by simulating/creating
    # missing indexes and unsimulating/dropping indexes
    # that exist but are not in the combination.
    def _prepare_cost_calculation(self, indexes, store_size=False):
        for index in set(indexes) - self.current_indexes:
            self._simulate_or_create_index(index, store_size=store_size)
        for index in self.current_indexes - set(indexes):
            self._unsimulate_or_drop_index(index)

        assert self.current_indexes == set(indexes)

    def _simulate_or_create_index(self, index, store_size=False):
        if self.cost_estimation == "whatif":
            self.what_if.simulate_index(index, store_size=store_size)
        elif self.cost_estimation == "actual_runtimes":
            self.db_connector.create_index(index)
        self.current_indexes.add(index)

    def _unsimulate_or_drop_index(self, index):
        if self.cost_estimation == "whatif":
            self.what_if.drop_simulated_index(index)
        elif self.cost_estimation == "actual_runtimes":
            self.db_connector.drop_index(index)
        self.current_indexes.remove(index)


    def _get_cost(self, query, relevant_indexes):
        
        if self.cost_estimation == "whatif":
            return self.db_connector.get_cost(query)
        elif self.cost_estimation == "actual_runtimes":
            # runtime, plan = self.db_connector.exec_query(query, timeout=max(orig_actual_runtimes, 1))
            
            runtime = self.collect_sample(query, relevant_indexes)
            
            return runtime
        
    def collect_sample(self, query, relevant_indexes):
        self.invoke_cnt += 1
        if self.cur_query is None or self.cur_query.text != query.text:
            self.cur_query = query
            proced_query_text = query.text.replace("\n"," ")
            print()
            logging.info(f'{query}: {proced_query_text}')
        orig_runtimes, orig_plan = self.sql2plan_tree[query.text]
        db_connector = self.db_connector
        ind_cfg = []
        for ind in relevant_indexes:
            ind_str = ind.table().name + "#" + ",".join([c.name for c in ind.columns])
            if ind.table().name in query.text:
                ind_cfg.append(ind_str)
        ind_cfg = tuple(sorted(ind_cfg))
        if len(ind_cfg) == 0:
            logging.info(f"invoke_cnt: {self.invoke_cnt} {(query,)} empty ind_cfg: {ind_cfg}")
            return orig_runtimes
        
        logic_cost, logic_plan = db_connector.exec_explain_query(query)
        orig_logic_plan = self.query2logic_plan[query]
        
        if are_dicts_equal(logic_plan, orig_logic_plan): 
            with_index_runtimes, with_index_plan = orig_runtimes, orig_plan
            plan_changed = False
        else:
            with_index_runtimes, with_index_plan = get_avg_runtimes(query, timeout=max(int(orig_runtimes)*20, 20), db_connector=db_connector)
            plan_changed = True
        self.all_samples.append((query, (tuple(), ind_cfg), (orig_runtimes, orig_plan), (with_index_runtimes, with_index_plan)))
        logging.info(f"invoke_cnt: {self.invoke_cnt} {(query, plan_changed, orig_runtimes, with_index_runtimes)} {ind_cfg}")
        if len(self.all_samples)%100 == 0:
            with open(self.breakpoint_path, 'wb') as f:
                pickle.dump(self.all_samples, f)
                logging.info(f"breakpoint invoke_cnt={self.invoke_cnt} save to {self.breakpoint_path}")
        return with_index_runtimes


    def complete_cost_estimation(self):
        self.completed = True

        for index in self.current_indexes.copy():
            self._unsimulate_or_drop_index(index)

        assert self.current_indexes == set()

    def _request_cache(self, query, indexes):
        q_i_hash = (query, frozenset(indexes))
        if q_i_hash in self.relevant_indexes_cache:
            relevant_indexes = self.relevant_indexes_cache[q_i_hash]
        else:
            relevant_indexes = self._relevant_indexes(query, indexes)
            self.relevant_indexes_cache[q_i_hash] = relevant_indexes

        # Check if query and corresponding relevant indexes in cache
        if (query, relevant_indexes) in self.cache:
            self.cache_hits += 1
            return self.cache[(query, relevant_indexes)]
        # If no cache hit request cost from database system
        else:
            cost = self._get_cost(query, relevant_indexes)
            self.cache[(query, relevant_indexes)] = cost
            return cost

    @staticmethod
    def _relevant_indexes(query, indexes):
        relevant_indexes = [
            x for x in indexes if any(c in query.columns for c in x.columns)
        ]
        return frozenset(relevant_indexes)
