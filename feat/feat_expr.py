from functools import lru_cache
import logging
import sys
from collections import deque

from feat.compute_selectivity import compute_column_selectivity
from feat.parse_expr import build_binary_node
from feat.feat_common_function import *

expr_compare_op2idx = {'PAD': 0, 'UNK': 1, '<>': 2, '~~': 3, '= ANY': 4, '>=': 5, '<=': 6, '>': 7, '<': 8, '=': 9, 'IS': 10, 'IS NOT': 11, '!~~': 12, '<> ALL': 13}
expr_logical_operator2idx = {"PAD": 0, "AND": 1, "OR": 2,  "UNK": 3}
index_utilized_cast_type = {"character varying": ["text",],
                            }
FILTER_MODE = "filter"
JOIN_MODE = "join"
INDEX_COND_MODE = "index_cond"


class TreeNode:
    def __init__(self):

        self.children = []
        self.parent = None
        self.feature = None
        self.pos = 0

    def add_child(self, tree_node):
        self.children.append(tree_node)

    def __str__(self):
        return 'feature {} with {} children'.format(self.feature, len(self.children))

    def __repr__(self):
        return self.__str__()


class ExprTreeEncoder:
    def __init__(self, db_stats=None, expr_encoder_mode=FILTER_MODE, enable_histogram=True):
        self.db_stats = db_stats
        self.index_col2order = None
        self.expr_encoder_mode = expr_encoder_mode
        self.alias2relation = None
        self.enable_histogram = enable_histogram

    def encode_expr(self, expr, index_col2order, index_tbl_name, plan_node):
        '''
        :param expr:
        :param index_col2order:
        :param index_tbl_name:
        :return:
        '''
        self.node_pos = -1
        self.index_tbl_name = index_tbl_name
        self.index_col2order = index_col2order
        self.tree_nodes = []
        self.plan_node = plan_node

        raw_expr_root = self.tran_expr_tree(expr)
        tree_root = self.traverse_expr_node(raw_expr_root)


        feat_dict = self.expr_tree2feat_dict(tree_root)

        self.tree_nodes.clear()
        del self.tree_nodes[:]

        return feat_dict

    def expr_tree2feat_dict(self, tree_node):
        adj_list, num_child, features = self.topo_sort(tree_node)
        features = np.array(features)

        if self.expr_encoder_mode != JOIN_MODE:
            buckets = np.linspace(0, 1, num=21)
            bucket_indices = np.digitize(features[:, 3], buckets)
            features[:, 3] = bucket_indices

        heights, shortest_path_mat = calculate_struct(adj_list, len(features))

        return {
            'x': features,
            'shortest_path_mat': shortest_path_mat,
            'heights': heights
        }

    @staticmethod
    def topo_sort(root_node):
        adj_list = []  # from parent to children
        num_child = []
        features = []

        to_visit = deque()
        to_visit.append((0, root_node))
        next_id = 1
        while to_visit:
            idx, node = to_visit.popleft()
            features.append(node.feature)
            num_child.append(len(node.children))
            for child in node.children:
                to_visit.append((next_id, child))
                adj_list.append((idx, next_id))
                next_id += 1

        return adj_list, num_child, features

    @lru_cache(10)
    def tran_expr_tree(self, expr_str):
        return build_binary_node(expr_str, 0)

    def set_alias2relation(self, alias2relation):
        self.alias2relation = alias2relation

    def calc_selectivity(self, predicate, col, compare_op):
        selectivity = -1 
        if not self.enable_histogram:
            return selectivity
        if '.' in col:
            col = col.split(".")[-1]
            
        if self.db_stats is not None and col in self.index_col2order and "ALL" not in compare_op and "ANY" not in compare_op:
            tbl_col = f"{self.index_tbl_name}.{col}"
            try:
                selectivity = compute_column_selectivity(tbl_col, predicate, self.db_stats)
            except Exception as e:
                selectivity = -1
        return selectivity

    def process_col_type(self, tbl_col_type):
        if "::" not in tbl_col_type:
            return tbl_col_type
        tbl_col, cast_type = tbl_col_type.split("::")
        col = tbl_col.split(".")[-1]
        col_key = f"{self.index_tbl_name}.{col}"
        # There are no statistics, and the default is to take advantage of the type conversion of the index
        if self.db_stats is None or col_key not in self.db_stats or "data_type" not in self.db_stats[col_key]:
            return tbl_col
        data_type = self.db_stats[col_key]["data_type"]
        if data_type not in index_utilized_cast_type:
            return tbl_col
        if cast_type in index_utilized_cast_type[data_type]:
            # Eliminate types
            return tbl_col
        logging.debug(f"Type conversion to index is not available, the original data type:{data_type} Conversion Goal Type:{cast_type} Check Items:{tbl_col} Index table name:{self.index_tbl_name}")
        return tbl_col

    def is_number(self, s):
        try:  
            float(s)
            return True
        except ValueError:  
            pass  
        try:
            import unicodedata  
            unicodedata.numeric(s)
            return True
        except (TypeError, ValueError):
            pass
        return False

    def encode_col_feat(self, tbl_col):
        
        if "Relation Name" in self.plan_node:
            col_tbl_name = self.plan_node["Relation Name"]
        elif "Alias" in self.plan_node:
            col_tbl_name = self.plan_node["Relation Name"]
        else:
            col_tbl_name = None
            
        return encode_col_feat(tbl_col, self.db_stats, self.index_tbl_name, self.alias2relation, self.index_col2order, tbl_name_of_col=col_tbl_name)
    

    def traverse_expr_node(self, expr_node):
        self.node_pos += 1

        logical_op = expr_node.logical_op
        predicate = expr_node.predicate
        # Initialization to -1 indicates that no selectivity is used
        selectivity = -1
        
        if logical_op is not None: # Logical operator encoding
            logical_op_code = expr_logical_operator2idx[logical_op] if logical_op in expr_logical_operator2idx else expr_logical_operator2idx["UNK"]
            if self.expr_encoder_mode == INDEX_COND_MODE:
                feature = [logical_op_code, None, None, selectivity, None] 
            else:
                feature = [logical_op_code, None, None, selectivity if self.expr_encoder_mode == FILTER_MODE else None]
                
        elif predicate is not None and not predicate.startswith('SubPlan ') and not predicate.startswith("NOT"):
            tokens = predicate.split(" ")
            left_col = self.process_col_type(tokens[0])
            right_col = tokens[-1]
            compare_op = f"{tokens[1]} {tokens[2]}" if len(tokens) > 2 and tokens[2] in {"ANY", "ALL"} else tokens[1]
            compare_op_code = expr_compare_op2idx[compare_op] if compare_op in expr_compare_op2idx else expr_compare_op2idx["UNK"]
            left_col_code = self.encode_col_feat(left_col)
            
            if self.expr_encoder_mode == INDEX_COND_MODE:
                
                feature = [None, left_col_code, compare_op_code, 
                           selectivity if is_col_name(right_col) else self.calc_selectivity(predicate, left_col, compare_op),
                           self.encode_col_feat(right_col) if is_col_name(right_col) else None]
            elif self.expr_encoder_mode == FILTER_MODE:
                feature = [None, left_col_code, compare_op_code, self.calc_selectivity(predicate, left_col, compare_op)]
            elif self.expr_encoder_mode == JOIN_MODE:
                feature = [None, left_col_code, compare_op_code, self.encode_col_feat(self.process_col_type(right_col))]
            else:
                raise Exception(f"unknow encoder mode: {self.expr_encoder_mode}")
            
        else:
            if self.expr_encoder_mode == INDEX_COND_MODE:
                feature = [None, None, None, None, selectivity]
            else:
                feature = [None, None, None, selectivity if self.expr_encoder_mode == FILTER_MODE else None]

        root = TreeNode()

        self.tree_nodes.append(root)

        root.feature = feature
        root.pos = self.node_pos

        if expr_node.left is not None:
            node = self.traverse_expr_node(expr_node.left)
            node.parent = root
            root.add_child(node)
        if expr_node.right is not None:
            node = self.traverse_expr_node(expr_node.right)
            node.parent = root
            root.add_child(node)

        return root
