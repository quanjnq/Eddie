import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
import logging
import sys
from functools import lru_cache
from collections import deque
from feat.feat_expr import ExprTreeEncoder, FILTER_MODE, JOIN_MODE,INDEX_COND_MODE
from feat.feat_common_function import *

ops_join_dict = {"Hash Join": 1, "Merge Join": 2, "Nested Loop": 3}
ops_sort_dict = {"Sort": 1}
ops_group_dict = {"Aggregate": 1, "Group": 2, "GroupAggregate": 3, "HashAggregate": 4}
ops_scan_dict = {"Seq Scan": 1, "Bitmap Heap Scan": 2, "Bitmap Index Scan": 3, "Index Scan": 4, "Index Only Scan": 5}
ops_index_scan_dict = {"Bitmap Index Scan": 3, "Index Scan": 4, "Index Only Scan": 5}
node_types2idx={'Unique':0,'Hash Join':1,'Bitmap Heap Scan':2,'Materialize':3,'SetOp':4,'Subquery Scan':5,'Aggregate':6,'BitmapAnd':7,'Gather Merge':8,'WindowAgg':9,'Sort':10,'Gather':11,'Index Scan':12,'Merge Join':13,'Bitmap Index Scan':14,'Nested Loop':15,'Index Only Scan':16,'CTE Scan':17,'Hash':18,'BitmapOr':19,'Limit':20,'Result':21,'Merge Append':22,'Append':23,'Group':24,'Seq Scan':25}
idx2node_type = {node_types2idx[k]: k for k in node_types2idx}
sort_order_consistency2idx = {"PAD": 0, "consistent":1, "inconsistent":2}
data_types2idx = {'character varying': 1, 'date': 2, 'time without time zone': 3, 'character': 4, 'integer': 5, 'numeric': 6, 'bigint': 7, 'double precision': 8, 'text': 9, 'bytea': 10, "smallint": 11}

class TreeNode:
    def __init__(self, node_type, type_id):
        self.node_type = node_type
        self.type_id = type_id
        self.children = []
        self.rounds = 0
        self.parent = None
        self.feature = None
        self.filter_dict = None
        self.index_cond_dict = None
        self.join_dict = None
        self.pos = 0
        self.sort_cols = []
        self.group_cols = []
        self.output_cols = []

    def add_child(self, tree_node):
        self.children.append(tree_node)

    def __str__(self):
        return '{} with {} children'.format(self.node_type, len(self.children))

    def __repr__(self):
        return self.__str__()


class PlanTreeEncoder:
    def __init__(self, db_stats=None, feat_params={}):
        
        # parameters
        # The maximum column length of a single index
        self.max_index_col_len = feat_params["max_index_col_num"] if "max_index_col_num" in feat_params else 3 # The default is 3 columns
        # A maximum of 5 columns of groups are supported
        self.max_gk_len = feat_params["max_group_col_num"] if "max_group_col_num" in feat_params else 5 # The default is 5 columns
        # A maximum of 5 columns of sort is supported
        self.max_sk_len = feat_params["max_sort_col_num"] if "max_sort_col_num" in feat_params else 5 # The default is 5 columns
        enable_histogram = feat_params["enable_histogram"] if "enable_histogram" in feat_params else True # Enabled by default

        # Expression encoders
        self.filter_expr_encoder = ExprTreeEncoder(db_stats=db_stats, expr_encoder_mode=FILTER_MODE, enable_histogram=enable_histogram)
        self.join_expr_encoder = ExprTreeEncoder(db_stats=db_stats, expr_encoder_mode=JOIN_MODE, enable_histogram=enable_histogram)
        self.index_cond_expr_encoder = ExprTreeEncoder(db_stats=db_stats, expr_encoder_mode=INDEX_COND_MODE, enable_histogram=enable_histogram)

        self.alias2relation = None
        self.db_stats = db_stats
        
    def set_alias2relation(self, alias2relation):
        self.alias2relation = alias2relation
        self.join_expr_encoder.set_alias2relation(alias2relation)
        self.filter_expr_encoder.set_alias2relation(alias2relation)
        self.index_cond_expr_encoder.set_alias2relation(alias2relation)

    def encode_plan_index(self, json_plan_root, index):
        '''
        Encoding Func
        :param json_plan_root: Query plan in JSON dictionary format
        :param index: 一条索引一条索引: tab#col1,col2
        :return: Encoded dictionary
        '''
        self.index_tbl_name = index.split("#")[0]
        cols = index.split("#")[1].split(",")
        # Indexed columns are mapped to sequential mapping
        self.index_col2order = dict()
        self.index_col2order["PAD"] = 0
        self.index_col2order["UNK"] = self.max_index_col_len + 1
        self.index_col2order["NOT_SAME_TBL"] = self.max_index_col_len + 2
        if len(cols) > self.max_index_col_len:
            logging.error(f"Index column length {len(cols)} Exceeding the set maximum length {self.max_index_col_len}, The corresponding index is:{index}, Truncated index column processing")
            cols = cols[:self.max_index_col_len]
        for pos, col in enumerate(cols):
            self.index_col2order[col] = pos + 1

        self.tree_nodes = []  # for mem collection
        self.node_pos = -1

        collated_dict = self.json_plan2feat_dict(json_plan_root)
        return collated_dict

    def json_plan2feat_dict(self, json_plan):
        tree_root = self.traverse_plan(json_plan)

        feat_dict = self.plan_tree2feat_dict(tree_root)

        self.tree_nodes.clear()
        del self.tree_nodes[:]

        return feat_dict

    def plan_tree2feat_dict(self, tree_root_node):

        adj_list, num_child, features, filter_dicts, filter_node_indices, index_cond_dicts, index_cond_node_indices, join_dicts, join_node_indices, sort_cols, group_cols, output_cols = self.topo_sort(tree_root_node)

        features = np.array(features)
        # Cost normalization treatment
        costmax = np.max(features[:, 1])
        costmin = np.min(features[:, 1])
        cardmax = np.max(features[:, 2])
        cardmin = np.min(features[:, 2])
        costabs = (costmax - costmin)
        cardabs = (cardmax - cardmin)
        if costabs == 0:
            features[:, 1] = 1
        else:
            features[:, 1] = (features[:, 1] - costmin) / costabs

        if cardabs == 0:
            features[:, 2] = 1
        else:
            features[:, 2] = (features[:, 2] - cardmin) / cardabs
        # Divided into 20 barrels
        buckets = np.linspace(0, 1, num=21)
        bucket_indices = np.digitize(features[:, 1], buckets)
        features[:, 1] = bucket_indices
        bucket_indices = np.digitize(features[:, 2], buckets)
        features[:, 2] = bucket_indices

        # Dimension filling and tree structure feature processing
        heights, shortest_path_mat = self.calculate_struct(tuple(adj_list), len(features))

        return {
            'x': features,
            'shortest_path_mat': shortest_path_mat,
            'heights': heights,
            "filter_dicts": filter_dicts,
            "filter_node_indices": filter_node_indices,
            "index_cond_dicts": index_cond_dicts,
            "index_cond_node_indices": index_cond_node_indices,
            "join_dicts": join_dicts,
            "join_node_indices": join_node_indices,
            "sort_cols": sort_cols,
            "group_cols": group_cols,
            "output_cols": output_cols,
        }

    @lru_cache(5)
    def calculate_struct(self, adj_list, node_num):
        return calculate_struct(adj_list, node_num)

    @staticmethod
    def topo_sort(root_node):
        adj_list = [] 
        num_child = []
        features = []
        filter_dicts = []
        filter_indices = []
        index_cond_dicts = []
        index_cond_indices = []
        join_dicts = []
        join_indices = []
        sort_cols = []
        group_cols = []
        output_cols = []
        
        to_visit = deque()
        to_visit.append((0, root_node))
        next_id = 1
        while to_visit:
            idx, node = to_visit.popleft()
            features.append(node.feature)
            sort_cols.append(node.sort_cols)
            group_cols.append(node.group_cols)
            output_cols.append(node.output_cols)
            if node.filter_dict is not None:
                filter_dicts.append(node.filter_dict)
                filter_indices.append(idx)
            if node.index_cond_dict is not None:
                index_cond_dicts.append(node.index_cond_dict)
                index_cond_indices.append(idx)
            if node.join_dict is not None:
                join_dicts.append(node.join_dict)
                join_indices.append(idx)
            num_child.append(len(node.children))
            for child in node.children:
                to_visit.append((next_id, child))
                adj_list.append((idx, next_id))
                next_id += 1

        return adj_list, num_child, features, filter_dicts, filter_indices, index_cond_dicts, index_cond_indices, join_dicts, join_indices, sort_cols, group_cols, output_cols

    def traverse_plan(self, plan):  # bfs accumulate plan
        self.node_pos += 1

        node_type = plan['Node Type']
        type_id = node_types2idx[node_type]

        root = TreeNode(node_type, type_id)

        self.tree_nodes.append(root)

        if 'Actual Rows' in plan:
            cost = plan["Actual Total Time"]
            card = plan["Actual Rows"]
        else:
            cost = plan['Total Cost']
            card = plan['Plan Rows']

        root.filter_dict = self.encode_filter(plan)
        root.index_cond_dict = self.encode_index_cond(plan)
        root.join_dict = self.encode_join(plan)
        root.pos = self.node_pos
        root.group_cols = self.encode_group(plan)
        sort_consistency, root.sort_cols = self.encode_sort(plan)
        root.output_cols = self.encode_output(plan)
        
        feature = np.array([type_id, cost, card, sort_consistency, card, cost])
        root.feature = feature

        if 'Plans' in plan:
            for subplan in plan['Plans']:
                subplan['parent'] = plan
                node = self.traverse_plan(subplan)
                node.parent = root
                root.add_child(node)
        return root
    
    def encode_index_cond(self, plan_node):
        # A filer exists and the index table is the same as the scan table
        if (plan_node['Node Type'] not in ops_index_scan_dict or 'Index Cond' not in plan_node):
            return None
        filter_expr = plan_node['Index Cond']

        return self.index_cond_expr_encoder.encode_expr(filter_expr, self.index_col2order, self.index_tbl_name, plan_node)

    def encode_filter(self, plan_node):
        if (plan_node['Node Type'] not in ops_scan_dict or 'Filter' not in plan_node):
            return None
        filter_expr = plan_node['Filter']

        return self.filter_expr_encoder.encode_expr(filter_expr, self.index_col2order, self.index_tbl_name, plan_node)

    def encode_join(self, plan_node):
        if plan_node['Node Type'] not in ops_join_dict:
            return None
        # Connection condition extraction
        if 'Hash Cond' in plan_node:
            join_cond_str = plan_node['Hash Cond']
        elif 'Merge Cond' in plan_node:
            join_cond_str = plan_node['Merge Cond']
        elif 'Join Filter' in plan_node:
            join_cond_str = plan_node['Join Filter']
        else:
            return None
        return self.join_expr_encoder.encode_expr(join_cond_str, self.index_col2order, self.index_tbl_name, plan_node)

    def encode_col_feat(self, tbl_col):
        return encode_col_feat(tbl_col, self.db_stats, self.index_tbl_name, self.alias2relation, self.index_col2order, tbl_name_of_col=None)
            
    def encode_group(self, plan_node):
        max_gk_len = self.max_gk_len
        if plan_node['Node Type'] not in ops_group_dict or 'Group Key' not in plan_node:
            return []

        vec = []
        group_keys = plan_node['Group Key']
        for pos, gk in enumerate(group_keys):
            vec.append(self.encode_col_feat(gk))

        return vec[:max_gk_len]
    
    def encode_output(self, plan_node):
        max_gk_len = self.max_gk_len
        if 'Output' not in plan_node:
            return []

        vec = []
        output_keys = plan_node['Output']
        
        for pos, ok in enumerate(output_keys):
            vec.append(self.encode_col_feat(ok))

        return vec[:max_gk_len]

    def encode_sort(self, plan_node):
        max_sk_len = self.max_sk_len
        if plan_node['Node Type'] not in ops_sort_dict or 'Sort Key' not in plan_node:
            return sort_order_consistency2idx["PAD"], []

        vec = []
        sort_keys = plan_node['Sort Key']
        sort_col_num = len(sort_keys)
        
        order_consistent_types = []
        for pos, sk in enumerate(sort_keys):
            if sk.endswith(" DESC"):
                order_consistent_types.append("DESC")
            if sk.endswith(" DESC") or sk.endswith(" ASC"):
                sk = sk.split(" ")[0]
            vec.append(self.encode_col_feat(sk))
                
        # Because index columns are always ASC sorted, they are simple to implement
        sort_consistency_val = sort_order_consistency2idx["inconsistent"] if (len(order_consistent_types) > 0) else sort_order_consistency2idx["consistent"]

        return sort_consistency_val, vec[:max_sk_len]


def encode_plan_index_config(encoder: PlanTreeEncoder, plan_root: dict, index_config:list, label:float):
    '''
    :param encoder:
    :param plan_root: json_dict
    :param index_config: ["customer#c_mktsegment,c_custkey", "orders#o_orderdate", "lineitem#l_orderkey,l_shipdate"]
    :param label: 0-1
    :return: dict
    '''
    # Aliases map to real relationships
    alias2relation = dict()
    relations = set()
    stack = [plan_root]
    while stack:
        cnode = stack.pop()
        if "Relation Name" in cnode and 'Alias' in cnode:
            alias2relation[cnode['Alias']] = cnode["Relation Name"]
            relations.add(cnode["Relation Name"])
        if "Plans" in cnode:
            for nod in cnode["Plans"]:
                stack.append(nod)
    # Set up alias mappings
    encoder.set_alias2relation(alias2relation)

    feat_dict = dict()
    struct_dict = dict()
    # Format conversion
    for index in index_config:
        # The query doesn't involve an index table
        if index.split("#")[0] not in relations:
            continue
        index_feat = dict()
        collated_dict = encoder.encode_plan_index(plan_root, index)
        for i, (row, sort_cols, group_cols, output_cols) in enumerate(zip(collated_dict["x"].astype(int).tolist(), collated_dict['sort_cols'], collated_dict['group_cols'], collated_dict['output_cols'])):
            index_feat[i] = {"operator_name":idx2node_type[row[0]], "operator":row[0], "cost":row[1], "rows":row[2], "sort_consistent":row[3], "raw_rows": row[4], "raw_cost": row[5],
                             "group_cols":group_cols, "sort_cols":sort_cols, "output_cols":output_cols, "filter_dict":dict(), "index_cond_dict":dict(), "join_dict":dict()}
            
        # filter condition
        for pos, collated_filt in zip(collated_dict["filter_node_indices"], collated_dict["filter_dicts"]):
            filter_dict = dict()
            for i, row in enumerate(collated_filt["x"].tolist()):
                filter_dict[i] = {"logical_operator": row[0], "left_col":row[1], "compare_operator":row[2], "selectivity":row[3]}
            index_feat[pos]["filter_dict"] = filter_dict
            
        # index condition
        for pos, collated_index_cond in zip(collated_dict["index_cond_node_indices"], collated_dict["index_cond_dicts"]):
            index_cond_dict = dict()
            for i, row in enumerate(collated_index_cond["x"].tolist()):
                index_cond_dict[i] = {"logical_operator": row[0], "left_col":row[1], "compare_operator":row[2], "selectivity":row[3], "right_col":row[4]}
            index_feat[pos]["index_cond_dict"] = index_cond_dict
        
        # join condition
        for pos, collated_join in zip(collated_dict["join_node_indices"], collated_dict["join_dicts"]):
            join_dict = dict()
            for i, row in enumerate(collated_join["x"].tolist()):
                join_dict[i] = {"logical_operator": row[0], "left_col":row[1], "compare_operator":row[2], "right_col":row[3]}
            index_feat[pos]["join_dict"] = join_dict
            
        feat_dict[index] = index_feat
        if len(struct_dict) == 0:
            filter_dict = dict()
            for pos, collated_filt in zip(collated_dict["filter_node_indices"], collated_dict["filter_dicts"]):
                filter_dict[pos] = {
                                    "shortest_path_mat": collated_filt["shortest_path_mat"].tolist(),
                                    "heights": collated_filt["heights"].tolist(),
                                    }
            join_dict = dict()
            for pos, collated_join in zip(collated_dict["join_node_indices"], collated_dict["join_dicts"]):
                join_dict[pos] = {
                                  "shortest_path_mat": collated_join["shortest_path_mat"].tolist(),
                                  "heights": collated_join["heights"].tolist(),
                                  }
            index_cond_dict = dict()
            for pos, collated_index_cond in zip(collated_dict["index_cond_node_indices"], collated_dict["index_cond_dicts"]):
                index_cond_dict[pos] = {
                                  "shortest_path_mat": collated_index_cond["shortest_path_mat"].tolist(),
                                  "heights": collated_index_cond["heights"].tolist(),
                                  }
            
            struct_dict = {
                           "shortest_path_mat": collated_dict["shortest_path_mat"].tolist(),
                           "heights": collated_dict["heights"].tolist(),
                           "filter_dict": filter_dict,
                           "join_dict": join_dict,
                           "index_cond_dict": index_cond_dict
                           }

    data_dict = {"feat": feat_dict, "struct": struct_dict, "label": label}
    return data_dict


if __name__ == "__main__":
    # Test Case
    exam_plan_tree = {'Node Type': 'Seq Scan', 'Parallel Aware': False, 'Relation Name': 'promotion', 'Schema': 'public', 'Alias': 'promotion', 'Startup Cost': 0.0, 'Total Cost': 19.25, 'Plan Rows': 9, 'Plan Width': 4, 'Output': ['p_start_date_sk'], 'Filter': '(promotion.p_promo_sk > 300)'}
    exam_plan_tree = {'Node Type': 'Sort', 'Parallel Aware': False, 'Startup Cost': 2461.09, 'Total Cost': 2462.34, 'Plan Rows': 500, 'Plan Width': 8, 'Output': ['promotion.p_start_date_sk', 'promotion.p_item_sk'], 'Sort Key': ['promotion.p_item_sk'], 'Plans': [{'Node Type': 'Hash Join', 'Parent Relationship': 'Outer', 'Parallel Aware': False, 'Join Type': 'Inner', 'Startup Cost': 24.25, 'Total Cost': 2438.67, 'Plan Rows': 500, 'Plan Width': 8, 'Output': ['promotion.p_start_date_sk', 'promotion.p_item_sk'], 'Inner Unique': False, 'Hash Cond': '(date_dim.d_date_sk = promotion.p_promo_sk)', 'Plans': [{'Node Type': 'Seq Scan', 'Parent Relationship': 'Outer', 'Parallel Aware': False, 'Relation Name': 'date_dim', 'Schema': 'public', 'Alias': 'date_dim', 'Startup Cost': 0.0, 'Total Cost': 2135.49, 'Plan Rows': 73049, 'Plan Width': 4, 'Output': ['date_dim.d_date_sk', 'date_dim.d_date_id', 'date_dim.d_date', 'date_dim.d_month_seq', 'date_dim.d_week_seq', 'date_dim.d_quarter_seq', 'date_dim.d_year', 'date_dim.d_dow', 'date_dim.d_moy', 'date_dim.d_dom', 'date_dim.d_qoy', 'date_dim.d_fy_year', 'date_dim.d_fy_quarter_seq', 'date_dim.d_fy_week_seq', 'date_dim.d_day_name', 'date_dim.d_quarter_name', 'date_dim.d_holiday', 'date_dim.d_weekend', 'date_dim.d_following_holiday', 'date_dim.d_first_dom', 'date_dim.d_last_dom', 'date_dim.d_same_day_ly', 'date_dim.d_same_day_lq', 'date_dim.d_current_day', 'date_dim.d_current_week', 'date_dim.d_current_month', 'date_dim.d_current_quarter', 'date_dim.d_current_year']}, {'Node Type': 'Hash', 'Parent Relationship': 'Inner', 'Parallel Aware': False, 'Startup Cost': 18.0, 'Total Cost': 18.0, 'Plan Rows': 500, 'Plan Width': 12, 'Output': ['promotion.p_start_date_sk', 'promotion.p_item_sk', 'promotion.p_promo_sk'], 'Plans': [{'Node Type': 'Seq Scan', 'Parent Relationship': 'Outer', 'Parallel Aware': False, 'Relation Name': 'promotion', 'Schema': 'public', 'Alias': 'promotion', 'Startup Cost': 0.0, 'Total Cost': 18.0, 'Plan Rows': 500, 'Plan Width': 12, 'Output': ['promotion.p_start_date_sk', 'promotion.p_item_sk', 'promotion.p_promo_sk']}]}]}]}
    exam_plan_tree = {'Node Type': 'Sort', 'Parallel Aware': False, 'Startup Cost': 2333.99, 'Total Cost': 2334.0, 'Plan Rows': 1, 'Plan Width': 8, 'Output': ['promotion.p_start_date_sk', 'promotion.p_item_sk'], 'Sort Key': ['promotion.p_item_sk'], 'Plans': [{'Node Type': 'Nested Loop', 'Parent Relationship': 'Outer', 'Parallel Aware': False, 'Join Type': 'Inner', 'Startup Cost': 0.02, 'Total Cost': 2333.98, 'Plan Rows': 1, 'Plan Width': 8, 'Output': ['promotion.p_start_date_sk', 'promotion.p_item_sk'], 'Inner Unique': False, 'Join Filter': '(date_dim.d_date_sk = promotion.p_promo_sk)', 'Plans': [{'Node Type': 'Index Scan', 'Parent Relationship': 'Outer', 'Parallel Aware': False, 'Scan Direction': 'Forward', 'Index Name': '<12697>btree_promotion_p_promo_sk_p_start_date_sk', 'Relation Name': 'promotion', 'Schema': 'public', 'Alias': 'promotion', 'Startup Cost': 0.02, 'Total Cost': 15.78, 'Plan Rows': 1, 'Plan Width': 12, 'Output': ['promotion.p_promo_sk', 'promotion.p_promo_id', 'promotion.p_start_date_sk', 'promotion.p_end_date_sk', 'promotion.p_item_sk', 'promotion.p_cost', 'promotion.p_response_target', 'promotion.p_promo_name', 'promotion.p_channel_dmail', 'promotion.p_channel_email', 'promotion.p_channel_catalog', 'promotion.p_channel_tv', 'promotion.p_channel_radio', 'promotion.p_channel_press', 'promotion.p_channel_event', 'promotion.p_channel_demo', 'promotion.p_channel_details', 'promotion.p_purpose', 'promotion.p_discount_active'], 'Index Cond': '(promotion.p_start_date_sk = 10)'}, {'Node Type': 'Seq Scan', 'Parent Relationship': 'Inner', 'Parallel Aware': False, 'Relation Name': 'date_dim', 'Schema': 'public', 'Alias': 'date_dim', 'Startup Cost': 0.0, 'Total Cost': 2318.11, 'Plan Rows': 7, 'Plan Width': 4, 'Output': ['date_dim.d_date_sk', 'date_dim.d_date_id', 'date_dim.d_date', 'date_dim.d_month_seq', 'date_dim.d_week_seq', 'date_dim.d_quarter_seq', 'date_dim.d_year', 'date_dim.d_dow', 'date_dim.d_moy', 'date_dim.d_dom', 'date_dim.d_qoy', 'date_dim.d_fy_year', 'date_dim.d_fy_quarter_seq', 'date_dim.d_fy_week_seq', 'date_dim.d_day_name', 'date_dim.d_quarter_name', 'date_dim.d_holiday', 'date_dim.d_weekend', 'date_dim.d_following_holiday', 'date_dim.d_first_dom', 'date_dim.d_last_dom', 'date_dim.d_same_day_ly', 'date_dim.d_same_day_lq', 'date_dim.d_current_day', 'date_dim.d_current_week', 'date_dim.d_current_month', 'date_dim.d_current_quarter', 'date_dim.d_current_year'], 'Filter': '(date_dim.d_date_sk < 10)'}]}]}
    exam_plan_tree = {'Node Type': 'Sort', 'Parallel Aware': False, 'Startup Cost': 2326.33, 'Total Cost': 2326.34, 'Plan Rows': 1, 'Plan Width': 8, 'Output': ['promotion.p_start_date_sk', 'promotion.p_item_sk'], 'Sort Key': ['promotion.p_item_sk'], 'Plans': [{'Node Type': 'Nested Loop', 'Parent Relationship': 'Outer', 'Parallel Aware': False, 'Join Type': 'Inner', 'Startup Cost': 0.02, 'Total Cost': 2326.32, 'Plan Rows': 1, 'Plan Width': 8, 'Output': ['promotion.p_start_date_sk', 'promotion.p_item_sk'], 'Inner Unique': False, 'Join Filter': '(date_dim.d_date_sk = promotion.p_promo_sk)', 'Plans': [{'Node Type': 'Index Scan', 'Parent Relationship': 'Outer', 'Parallel Aware': False, 'Scan Direction': 'Forward', 'Index Name': '<12697>btree_promotion_p_promo_sk_p_start_date_sk', 'Relation Name': 'promotion', 'Schema': 'public', 'Alias': 'promotion', 'Startup Cost': 0.02, 'Total Cost': 8.12, 'Plan Rows': 1, 'Plan Width': 12, 'Output': ['promotion.p_promo_sk', 'promotion.p_promo_id', 'promotion.p_start_date_sk', 'promotion.p_end_date_sk', 'promotion.p_item_sk', 'promotion.p_cost', 'promotion.p_response_target', 'promotion.p_promo_name', 'promotion.p_channel_dmail', 'promotion.p_channel_email', 'promotion.p_channel_catalog', 'promotion.p_channel_tv', 'promotion.p_channel_radio', 'promotion.p_channel_press', 'promotion.p_channel_event', 'promotion.p_channel_demo', 'promotion.p_channel_details', 'promotion.p_purpose', 'promotion.p_discount_active'], 'Index Cond': '((promotion.p_promo_sk < 10) AND (promotion.p_start_date_sk = 10))'}, {'Node Type': 'Seq Scan', 'Parent Relationship': 'Inner', 'Parallel Aware': False, 'Relation Name': 'date_dim', 'Schema': 'public', 'Alias': 'date_dim', 'Startup Cost': 0.0, 'Total Cost': 2318.11, 'Plan Rows': 7, 'Plan Width': 4, 'Output': ['date_dim.d_date_sk', 'date_dim.d_date_id', 'date_dim.d_date', 'date_dim.d_month_seq', 'date_dim.d_week_seq', 'date_dim.d_quarter_seq', 'date_dim.d_year', 'date_dim.d_dow', 'date_dim.d_moy', 'date_dim.d_dom', 'date_dim.d_qoy', 'date_dim.d_fy_year', 'date_dim.d_fy_quarter_seq', 'date_dim.d_fy_week_seq', 'date_dim.d_day_name', 'date_dim.d_quarter_name', 'date_dim.d_holiday', 'date_dim.d_weekend', 'date_dim.d_following_holiday', 'date_dim.d_first_dom', 'date_dim.d_last_dom', 'date_dim.d_same_day_ly', 'date_dim.d_same_day_lq', 'date_dim.d_current_day', 'date_dim.d_current_week', 'date_dim.d_current_month', 'date_dim.d_current_quarter', 'date_dim.d_current_year'], 'Filter': '(date_dim.d_date_sk < 10)'}]}]}
    exam_plan_tree = {'Node Type': 'Aggregate', 'Strategy': 'Plain', 'Partial Mode': 'Finalize', 'Parallel Aware': False, 'Startup Cost': 363766.96, 'Total Cost': 363766.97, 'Plan Rows': 1, 'Plan Width': 96, 'Output': ['min((cn.name)::text)', 'min((mi_idx.info)::text)', 'min((t.title)::text)'], 'Plans': [{'Node Type': 'Gather', 'Parent Relationship': 'Outer', 'Parallel Aware': False, 'Startup Cost': 363766.84, 'Total Cost': 363766.95, 'Plan Rows': 1, 'Plan Width': 96, 'Output': ['(PARTIAL min((cn.name)::text))', '(PARTIAL min((mi_idx.info)::text))', '(PARTIAL min((t.title)::text))'], 'Workers Planned': 1, 'Single Copy': False, 'Plans': [{'Node Type': 'Aggregate', 'Strategy': 'Plain', 'Partial Mode': 'Partial', 'Parent Relationship': 'Outer', 'Parallel Aware': False, 'Startup Cost': 362766.84, 'Total Cost': 362766.85, 'Plan Rows': 1, 'Plan Width': 96, 'Output': ['PARTIAL min((cn.name)::text)', 'PARTIAL min((mi_idx.info)::text)', 'PARTIAL min((t.title)::text)'], 'Plans': [{'Node Type': 'Hash Join', 'Parent Relationship': 'Outer', 'Parallel Aware': True, 'Join Type': 'Inner', 'Startup Cost': 357854.04, 'Total Cost': 362766.83, 'Plan Rows': 1, 'Plan Width': 41, 'Output': ['cn.name', 'mi_idx.info', 't.title'], 'Inner Unique': False, 'Hash Cond': '(cn.id = mc.company_id)', 'Plans': [{'Node Type': 'Seq Scan', 'Parent Relationship': 'Outer', 'Parallel Aware': True, 'Relation Name': 'company_name', 'Schema': 'public', 'Alias': 'cn', 'Startup Cost': 0.0, 'Total Cost': 4722.92, 'Plan Rows': 50631, 'Plan Width': 23, 'Output': ['cn.id', 'cn.name', 'cn.country_code', 'cn.imdb_id', 'cn.name_pcode_nf', 'cn.name_pcode_sf', 'cn.md5sum'], 'Filter': "((cn.country_code)::text = '[us]'::text)"}, {'Node Type': 'Hash', 'Parent Relationship': 'Inner', 'Parallel Aware': True, 'Startup Cost': 357854.01, 'Total Cost': 357854.01, 'Plan Rows': 2, 'Plan Width': 26, 'Output': ['mc.company_id', 'mi_idx.info', 't.title'], 'Plans': [{'Node Type': 'Hash Join', 'Parent Relationship': 'Outer', 'Parallel Aware': False, 'Join Type': 'Inner', 'Startup Cost': 324115.81, 'Total Cost': 357854.01, 'Plan Rows': 2, 'Plan Width': 26, 'Output': ['mc.company_id', 'mi_idx.info', 't.title'], 'Inner Unique': False, 'Hash Cond': '(mc.company_type_id = ct.id)', 'Plans': [{'Node Type': 'Hash Join', 'Parent Relationship': 'Outer', 'Parallel Aware': True, 'Join Type': 'Inner', 'Startup Cost': 324114.75, 'Total Cost': 357852.89, 'Plan Rows': 10, 'Plan Width': 30, 'Output': ['mc.company_type_id', 'mc.company_id', 'mi_idx.info', 't.title'], 'Inner Unique': False, 'Hash Cond': '(mc.movie_id = t.id)', 'Plans': [{'Node Type': 'Seq Scan', 'Parent Relationship': 'Outer', 'Parallel Aware': True, 'Relation Name': 'movie_companies', 'Schema': 'public', 'Alias': 'mc', 'Startup Cost': 0.0, 'Total Cost': 29661.37, 'Plan Rows': 1087137, 'Plan Width': 12, 'Output': ['mc.id', 'mc.movie_id', 'mc.company_id', 'mc.company_type_id', 'mc.note']}, {'Node Type': 'Hash', 'Parent Relationship': 'Inner', 'Parallel Aware': True, 'Startup Cost': 324114.72, 'Total Cost': 324114.72, 'Plan Rows': 2, 'Plan Width': 34, 'Output': ['mi.movie_id', 'mi_idx.info', 'mi_idx.movie_id', 't.title', 't.id'], 'Plans': [{'Node Type': 'Hash Join', 'Parent Relationship': 'Outer', 'Parallel Aware': True, 'Join Type': 'Inner', 'Startup Cost': 270682.94, 'Total Cost': 324114.72, 'Plan Rows': 2, 'Plan Width': 34, 'Output': ['mi.movie_id', 'mi_idx.info', 'mi_idx.movie_id', 't.title', 't.id'], 'Inner Unique': False, 'Hash Cond': '(t.id = mi.movie_id)', 'Plans': [{'Node Type': 'Seq Scan', 'Parent Relationship': 'Outer', 'Parallel Aware': True, 'Relation Name': 'title', 'Schema': 'public', 'Alias': 't', 'Startup Cost': 0.0, 'Total Cost': 51799.95, 'Plan Rows': 435151, 'Plan Width': 21, 'Output': ['t.id', 't.title', 't.imdb_index', 't.kind_id', 't.production_year', 't.imdb_id', 't.phonetic_code', 't.episode_of_id', 't.season_nr', 't.episode_nr', 't.series_years', 't.md5sum'], 'Filter': '((t.production_year >= 2000) AND (t.production_year <= 2010))'}, {'Node Type': 'Hash', 'Parent Relationship': 'Inner', 'Parallel Aware': True, 'Startup Cost': 270682.88, 'Total Cost': 270682.88, 'Plan Rows': 5, 'Plan Width': 13, 'Output': ['mi.movie_id', 'mi_idx.info', 'mi_idx.movie_id'], 'Plans': [{'Node Type': 'Hash Join', 'Parent Relationship': 'Outer', 'Parallel Aware': False, 'Join Type': 'Inner', 'Startup Cost': 15442.6, 'Total Cost': 270682.88, 'Plan Rows': 5, 'Plan Width': 13, 'Output': ['mi.movie_id', 'mi_idx.info', 'mi_idx.movie_id'], 'Inner Unique': False, 'Hash Cond': '(mi.info_type_id = it1.id)', 'Plans': [{'Node Type': 'Hash Join', 'Parent Relationship': 'Outer', 'Parallel Aware': True, 'Join Type': 'Inner', 'Startup Cost': 15440.17, 'Total Cost': 270678.54, 'Plan Rows': 501, 'Plan Width': 17, 'Output': ['mi_idx.info', 'mi_idx.movie_id', 'mi.movie_id', 'mi.info_type_id'], 'Inner Unique': False, 'Hash Cond': '(mi.movie_id = mi_idx.movie_id)', 'Plans': [{'Node Type': 'Seq Scan', 'Parent Relationship': 'Outer', 'Parallel Aware': True, 'Relation Name': 'movie_info', 'Schema': 'public', 'Alias': 'mi', 'Startup Cost': 0.0, 'Total Cost': 254716.25, 'Plan Rows': 138673, 'Plan Width': 8, 'Output': ['mi.id', 'mi.movie_id', 'mi.info_type_id', 'mi.info', 'mi.note'], 'Filter': "((mi.info)::text = ANY ('{Drama,Horror,Western,Family}'::text[]))"}, {'Node Type': 'Hash', 'Parent Relationship': 'Inner', 'Parallel Aware': True, 'Startup Cost': 15431.35, 'Total Cost': 15431.35, 'Plan Rows': 706, 'Plan Width': 9, 'Output': ['mi_idx.info', 'mi_idx.movie_id'], 'Plans': [{'Node Type': 'Hash Join', 'Parent Relationship': 'Outer', 'Parallel Aware': False, 'Join Type': 'Inner', 'Startup Cost': 2.43, 'Total Cost': 15431.35, 'Plan Rows': 706, 'Plan Width': 9, 'Output': ['mi_idx.info', 'mi_idx.movie_id'], 'Inner Unique': False, 'Hash Cond': '(mi_idx.info_type_id = it2.id)', 'Plans': [{'Node Type': 'Seq Scan', 'Parent Relationship': 'Outer', 'Parallel Aware': True, 'Relation Name': 'movie_info_idx', 'Schema': 'public', 'Alias': 'mi_idx', 'Startup Cost': 0.0, 'Total Cost': 15122.68, 'Plan Rows': 79782, 'Plan Width': 13, 'Output': ['mi_idx.id', 'mi_idx.movie_id', 'mi_idx.info_type_id', 'mi_idx.info', 'mi_idx.note'], 'Filter': "((mi_idx.info)::text > '7.0'::text)"}, {'Node Type': 'Hash', 'Parent Relationship': 'Inner', 'Parallel Aware': False, 'Startup Cost': 2.41, 'Total Cost': 2.41, 'Plan Rows': 1, 'Plan Width': 4, 'Output': ['it2.id'], 'Plans': [{'Node Type': 'Seq Scan', 'Parent Relationship': 'Outer', 'Parallel Aware': False, 'Relation Name': 'info_type', 'Schema': 'public', 'Alias': 'it2', 'Startup Cost': 0.0, 'Total Cost': 2.41, 'Plan Rows': 1, 'Plan Width': 4, 'Output': ['it2.id'], 'Filter': "((it2.info)::text = 'rating'::text)"}]}]}]}]}, {'Node Type': 'Hash', 'Parent Relationship': 'Inner', 'Parallel Aware': False, 'Startup Cost': 2.41, 'Total Cost': 2.41, 'Plan Rows': 1, 'Plan Width': 4, 'Output': ['it1.id'], 'Plans': [{'Node Type': 'Seq Scan', 'Parent Relationship': 'Outer', 'Parallel Aware': False, 'Relation Name': 'info_type', 'Schema': 'public', 'Alias': 'it1', 'Startup Cost': 0.0, 'Total Cost': 2.41, 'Plan Rows': 1, 'Plan Width': 4, 'Output': ['it1.id'], 'Filter': "((it1.info)::text = 'genres'::text)"}]}]}]}]}]}]}, {'Node Type': 'Hash', 'Parent Relationship': 'Inner', 'Parallel Aware': False, 'Startup Cost': 1.05, 'Total Cost': 1.05, 'Plan Rows': 1, 'Plan Width': 4, 'Output': ['ct.id'], 'Plans': [{'Node Type': 'Seq Scan', 'Parent Relationship': 'Outer', 'Parallel Aware': False, 'Relation Name': 'company_type', 'Schema': 'public', 'Alias': 'ct', 'Startup Cost': 0.0, 'Total Cost': 1.05, 'Plan Rows': 1, 'Plan Width': 4, 'Output': ['ct.id'], 'Filter': "((ct.kind)::text = 'production companies'::text)"}]}]}]}]}]}]}]}
    exam_ind_conf =  ['catalog_page#cp_catalog_page_sk', 'catalog_sales#cs_sold_date_sk', 'promotion#p_promo_sk', 'web_sales#ws_item_sk', 'item#i_color,i_manufact']
    exam_ind_conf =  ['cast_info#movie_id,note', 'company_name#country_code']
    db_fp = "./db_stats_data/indexselection_tpcds___10_stats.json"
    db_fp = "./db_stats_data/imdbload_stats.json"
    with open(db_fp) as f:
        db_stst = json.load(f)
    print(type(exam_plan_tree), exam_plan_tree['Node Type'])
    data_feat = encode_plan_index_config(PlanTreeEncoder(db_stst), exam_plan_tree, exam_ind_conf, label=0)
    print("===>")
    print(json.dumps(data_feat))
