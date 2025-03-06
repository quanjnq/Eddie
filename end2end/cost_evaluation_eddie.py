from torch.utils.data import DataLoader

from trainer.trainer import get_batch_data_to_device
from util.draw_util import *
from eddie.eddie_dataset import *
from feat.feat_plan import encode_plan_index_config, PlanTreeEncoder
from run_eddie import ModelArgs
from selection.what_if_index_creation import WhatIfIndexCreation

class CostEvaluation:
    def __init__(self, db_connector, cost_estimation="whatif", db_stat=None, model=None, sql2plan=None):
        self.db_connector = db_connector
        self.cost_estimation = cost_estimation
        self.what_if = WhatIfIndexCreation(db_connector)
        self.current_indexes = set()
        self.cost_requests = 0
        self.cache_hits = 0
        # Cache structure:
        # {(query_object, relevant_indexes): cost}
        self.cache = {}
        self.completed = False
        # It is not necessary to drop hypothetical indexes during __init__().
        # These are only created per connection. Hence, non should be present.

        self.relevant_indexes_cache = {}
        
        self.db_stat = db_stat
        self.model = model
        self.sql2plan_tree = sql2plan if sql2plan else {}
        
    def estimate_size(self, index):
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


    def _get_cost(self, query, indexes):
        if query.text in self.sql2plan_tree:
            cost, plan = self.sql2plan_tree[query.text]
        else:
            cost, plan = self.db_connector.exec_query(query)
            self.sql2plan_tree[query.text] = (cost, plan)
            
        ind_cfg = []
        for ind in indexes:
            ind_str = ind.table().name + "#" + ",".join([c.name for c in ind.columns])
            if ind.table().name in query.text:
                ind_cfg.append(ind_str)
        if len(ind_cfg) == 0:
            return cost
        try:
            benefit = self.get_benefit(plan, ind_cfg, )
            if benefit < 0.02: # Low-yield pre-treatment
                benefit = 0
        except Exception as e:
            benefit = 0
        pred_cost = (1 - benefit) * cost # cost restore
        return pred_cost


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
    
    @staticmethod
    def infer(feat_dict, model, args: ModelArgs):
        val_ds = EddieDataset([feat_dict])

        dataCollator = EddieDataCollator(args.max_sort_col_num, args.max_output_col_num, args.max_attn_dist, args.log_label)
        val_dataloader = DataLoader(val_ds, batch_size=args.batch_size, collate_fn=dataCollator.collate_fn, num_workers=0, pin_memory=True)
        

        model.to(args.device)
        
        model.eval()
        with torch.no_grad():
            for batch in val_dataloader:
                features, label = get_batch_data_to_device(batch, args.device)

                val_preds = model(features)
                val_preds = val_preds.squeeze()
                
        val_pred = val_preds.cpu().detach().numpy().flatten().tolist()[0]
        if args.log_label:
            return min(math.pow(math.e, val_pred) - 1, 1.0)
        else:
            return val_pred


    def get_benefit(self, orig_plan, index_config):
        feat_params = {"max_index_col_num": 3, "max_group_col_num": 5, "max_sort_col_num": 5}
        encoder = PlanTreeEncoder(self.db_stat, feat_params)
        feat_dict = encode_plan_index_config(encoder, orig_plan, index_config, 0)
        benefit = self.infer(feat_dict, self.model)
        return benefit
