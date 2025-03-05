import torch
import math

from feat.feat_plan import sort_order_consistency2idx, data_types2idx
from torch.utils.data import Dataset


class EddieDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
    

class EddieDataCollator:
    def __init__(self, max_sort_col_num, max_output_col_num, max_attn_dist, log_label, \
                 predicate_types=['filter', 'join', 'index_cond']):
        self.max_sort_col_num = max_sort_col_num
        self.max_output_col_num = max_output_col_num
        self.max_attn_dist = max_attn_dist
        self.log_label = log_label
        self.predicate_types = predicate_types
        
    def collate_fn(self, batch):
        batch_size = len(batch)
        batch_feat = {}
        
        max_index_num = max(len(sample['feat']) for sample in batch)
        max_node_num = max(len(nodes) for sample in batch for index, nodes in sample['feat'].items())

        self.predicate_types = ['filter', 'join', 'index_cond']
        max_predicate_num = {}
        for pt in self.predicate_types:
            max_predicate_num[pt] = max(len(node[f'{pt}_dict']) for sample in batch for index, nodes in sample['feat'].items() for ni, node in nodes.items())
        
        # Initialize feature tensor
        batch_feat['operator'] = torch.zeros((batch_size, max_index_num, max_node_num), dtype=torch.int)
        batch_feat['cost'] = torch.zeros((batch_size, max_index_num, max_node_num), dtype=torch.int)
        batch_feat['rows'] = torch.zeros((batch_size, max_index_num, max_node_num), dtype=torch.int)

        batch_feat['sort_consistent'] = torch.zeros((batch_size, max_index_num, max_node_num), dtype=torch.int)
        batch_feat['sort_col_pos'] = torch.zeros((batch_size, max_index_num, max_node_num, self.max_sort_col_num), dtype=torch.int)
        batch_feat['sort_col_type'] = torch.zeros((batch_size, max_index_num, max_node_num, self.max_sort_col_num), dtype=torch.int)
        batch_feat['sort_col_dist_cnt'] = torch.zeros((batch_size, max_index_num, max_node_num, self.max_sort_col_num), dtype=torch.int)
        batch_feat['sort_col_dist_frac'] = torch.zeros((batch_size, max_index_num, max_node_num, self.max_sort_col_num), dtype=torch.float)
        batch_feat['sort_col_null_frac'] = torch.zeros((batch_size, max_index_num, max_node_num, self.max_sort_col_num), dtype=torch.float)
        
        batch_feat['output_col_pos'] = torch.zeros((batch_size, max_index_num, max_node_num, self.max_output_col_num), dtype=torch.int)
        batch_feat['output_col_type'] = torch.zeros((batch_size, max_index_num, max_node_num, self.max_output_col_num), dtype=torch.int)
        batch_feat['output_col_dist_cnt'] = torch.zeros((batch_size, max_index_num, max_node_num, self.max_output_col_num), dtype=torch.int)
        batch_feat['output_col_dist_frac'] = torch.zeros((batch_size, max_index_num, max_node_num, self.max_output_col_num), dtype=torch.float)
        batch_feat['output_col_null_frac'] = torch.zeros((batch_size, max_index_num, max_node_num, self.max_output_col_num), dtype=torch.float)
        
        batch_feat['node_heights'] = torch.zeros((batch_size, max_node_num), dtype=torch.int)
        batch_feat['node_rel_pos_mat'] = torch.zeros((batch_size, max_node_num+1, max_node_num+1), dtype=torch.int)
        batch_feat['node_tree_attn_bias'] = torch.zeros((batch_size, max_node_num+1, max_node_num+1), dtype=torch.float)
        batch_feat['index_attn_bias'] = torch.zeros((batch_size, max_node_num+1, max_index_num, max_index_num), dtype=torch.float)
        
        batch_feat['node_padding_mask'] = torch.zeros((batch_size, max_index_num, max_node_num), dtype=torch.int)
        batch_label = torch.zeros(batch_size, dtype=torch.float)
        
        # Pre-calculate the count of various predicate types
        for pt in self.predicate_types:
            predicate_tree_idx_mapping = torch.full((batch_size, max_index_num, max_node_num), -1, dtype=torch.int)
            predicate_tree_idx = 0

            for si, sample in enumerate(batch):
                for ii, (index, nodes) in enumerate(sample['feat'].items()):
                    for ni, node in nodes.items(): 
                        ni = int(ni)
                        if len(node[f'{pt}_dict']) > 0:
                            predicate_tree_idx_mapping[si, ii, ni] = predicate_tree_idx
                            predicate_tree_idx += 1
            
            batch_feat[f'{pt}_predicate_tree_mask'] = torch.where(predicate_tree_idx_mapping == -1, torch.tensor(0), torch.tensor(1))
            batch_feat[f'{pt}_predicate_tree_idx_mapping'] = predicate_tree_idx_mapping
            
            predicate_tree_cnt = predicate_tree_idx
            batch_feat[f'{pt}_logical_operator'] = torch.zeros((predicate_tree_cnt, max_predicate_num[pt]), dtype=torch.int)
            batch_feat[f'{pt}_compare_operator'] = torch.zeros((predicate_tree_cnt, max_predicate_num[pt]), dtype=torch.int)
            batch_feat[f'{pt}_selectivity'] = torch.zeros((predicate_tree_cnt, max_predicate_num[pt]), dtype=torch.int)
            batch_feat[f'{pt}_left_col_pos'] = torch.zeros((predicate_tree_cnt, max_predicate_num[pt]), dtype=torch.int)
            batch_feat[f'{pt}_left_col_type'] = torch.zeros((predicate_tree_cnt, max_predicate_num[pt]), dtype=torch.int)
            batch_feat[f'{pt}_left_col_dist_cnt'] = torch.zeros((predicate_tree_cnt, max_predicate_num[pt]), dtype=torch.int)
            batch_feat[f'{pt}_left_col_dist_frac'] = torch.zeros((predicate_tree_cnt, max_predicate_num[pt]), dtype=torch.float)
            batch_feat[f'{pt}_left_col_null_frac'] = torch.zeros((predicate_tree_cnt, max_predicate_num[pt]), dtype=torch.float)
            batch_feat[f'{pt}_right_col_pos'] = torch.zeros((predicate_tree_cnt, max_predicate_num[pt]), dtype=torch.int)
            batch_feat[f'{pt}_right_col_type'] = torch.zeros((predicate_tree_cnt, max_predicate_num[pt]), dtype=torch.int)
            batch_feat[f'{pt}_right_col_dist_cnt'] = torch.zeros((predicate_tree_cnt, max_predicate_num[pt]), dtype=torch.int)
            batch_feat[f'{pt}_right_col_dist_frac'] = torch.zeros((predicate_tree_cnt, max_predicate_num[pt]), dtype=torch.float)
            batch_feat[f'{pt}_right_col_null_frac'] = torch.zeros((predicate_tree_cnt, max_predicate_num[pt]), dtype=torch.float)
            
            batch_feat[f'{pt}_heights'] = torch.zeros((predicate_tree_cnt, max_predicate_num[pt]), dtype=torch.int)
            batch_feat[f'{pt}_rel_pos_mat'] = torch.zeros((predicate_tree_cnt, max_predicate_num[pt]+1, max_predicate_num[pt]+1), dtype=torch.int)
            batch_feat[f'{pt}_tree_attn_bias'] = torch.zeros((predicate_tree_cnt, max_predicate_num[pt]+1, max_predicate_num[pt]+1), dtype=torch.float)
        
        for si, sample in enumerate(batch):
            index_num = len(sample['feat'])
            for ii, (index, nodes) in enumerate(sample['feat'].items()):
                node_num = len(nodes)
                batch_feat['node_padding_mask'][si, ii, :node_num] = 1
                
                for ni, node in nodes.items(): 
                    ni = int(ni)
                    batch_feat['operator'][si, ii, ni] = node['operator']
                    batch_feat['cost'][si, ii, ni] = node['cost']
                    batch_feat['rows'][si, ii, ni] = node['rows']
                    
                    # Merge group cols into sort cols
                    sort_cols = node['group_cols'] if len(node['group_cols']) > 0 else node['sort_cols']
                    batch_feat['sort_consistent'][si, ii, ni] = sort_order_consistency2idx['consistent'] if len(node['group_cols']) > 0 else node['sort_consistent']
                    for ci, col in enumerate(sort_cols):
                        batch_feat['sort_col_pos'][si, ii, ni, ci] = col['index_pos']
                        batch_feat['sort_col_type'][si, ii, ni, ci] = data_types2idx[col['data_type']] if col['data_type'] else 0
                        batch_feat['sort_col_dist_cnt'][si, ii, ni, ci] = math.log(1 +col['dist_cnt']) if col['dist_cnt'] else 0
                        batch_feat['sort_col_dist_frac'][si, ii, ni, ci] = col['dist_frac'] if col['dist_frac'] else 0
                        batch_feat['sort_col_null_frac'][si, ii, ni, ci] = col['null_frac'] if col['null_frac'] else 0
                    
                    for ci, col in enumerate(node['output_cols']):
                        batch_feat['output_col_pos'][si, ii, ni, ci] = col['index_pos']
                        batch_feat['output_col_type'][si, ii, ni, ci] = data_types2idx[col['data_type']] if col['data_type'] else 0
                        batch_feat['output_col_dist_cnt'][si, ii, ni, ci] = math.log(1 +col['dist_cnt']) if col['dist_cnt'] else 0
                        batch_feat['output_col_dist_frac'][si, ii, ni, ci] = col['dist_frac'] if col['dist_frac'] else 0
                        batch_feat['output_col_null_frac'][si, ii, ni, ci] = col['null_frac'] if col['null_frac'] else 0
                        
                    for pt in self.predicate_types:
                        for pi, predicate in node[f'{pt}_dict'].items():
                            pi = int(pi)
                            predicate_tree_idx = batch_feat[f'{pt}_predicate_tree_idx_mapping'][si, ii, ni]
                            
                            batch_feat[f'{pt}_logical_operator'][predicate_tree_idx, pi] = predicate['logical_operator'] if predicate['logical_operator'] else 0
                            batch_feat[f'{pt}_compare_operator'][predicate_tree_idx, pi] = predicate['compare_operator'] if predicate['compare_operator'] else 0
                            batch_feat[f'{pt}_selectivity'][predicate_tree_idx, pi] = predicate['selectivity'] if 'selectivity' in predicate else 0
                                
                            if predicate['left_col']:
                                left_col = predicate['left_col']
                                batch_feat[f'{pt}_left_col_pos'][predicate_tree_idx, pi] = left_col['index_pos'] if left_col['index_pos'] else 0
                                batch_feat[f'{pt}_left_col_type'][predicate_tree_idx, pi] = data_types2idx[left_col['data_type']] if left_col['data_type'] else 0
                                batch_feat[f'{pt}_left_col_dist_cnt'][predicate_tree_idx, pi] = math.log(1 +left_col['dist_cnt']) if left_col['dist_cnt'] else 0
                                batch_feat[f'{pt}_left_col_dist_frac'][predicate_tree_idx, pi] = left_col['dist_frac'] if left_col['dist_frac'] else 0
                                batch_feat[f'{pt}_left_col_null_frac'][predicate_tree_idx, pi] = left_col['null_frac'] if left_col['null_frac'] else 0
                            
                            if 'right_col' in predicate and predicate['right_col']:
                                right_col = predicate['right_col']
                                batch_feat[f'{pt}_right_col_pos'][predicate_tree_idx, pi] = right_col['index_pos'] if right_col['index_pos'] else 0
                                batch_feat[f'{pt}_right_col_type'][predicate_tree_idx, pi] = data_types2idx[right_col['data_type']] if right_col['data_type'] else 0
                                batch_feat[f'{pt}_right_col_dist_cnt'][predicate_tree_idx, pi] = math.log(1 +right_col['dist_cnt']) if right_col['dist_cnt'] else 0
                                batch_feat[f'{pt}_right_col_dist_frac'][predicate_tree_idx, pi] = right_col['dist_frac'] if right_col['dist_frac'] else 0
                                batch_feat[f'{pt}_right_col_null_frac'][predicate_tree_idx, pi] = right_col['null_frac'] if right_col['null_frac'] else 0
                        
            # Process the heights tensor for nodes
            node_heights = sample['struct']['heights']
            node_num = len(node_heights)
            batch_feat['node_heights'][si, :node_num] = torch.tensor(node_heights, dtype=torch.int)
            
            # Process the tree_attn_bias tensor for nodes
            node_shortest_path_mat = sample['struct']['shortest_path_mat']
            batch_feat['node_dist'] = torch.tensor(node_shortest_path_mat, dtype=torch.int)
            batch_feat['node_rel_pos_mat'][si, 1:node_num+1, 1:node_num+1] = batch_feat['node_dist']
            max_dist = min(node_num, self.max_attn_dist)
            batch_feat['node_tree_attn_bias'][si, 1:node_num+1, 1:node_num+1][batch_feat['node_dist'] >= max_dist] = float('-inf') # Treat as unreachable
            batch_feat['node_tree_attn_bias'][si, :, node_num+1:] = float('-inf')

            batch_feat['index_attn_bias'][si, :, :, index_num:] = float('-inf')

            # Process the heights and tree_attn_bias tensors for predicates
            for pt in self.predicate_types:
                predicate_struct_dict = sample['struct'][f'{pt}_dict']
                for ni, predicates in predicate_struct_dict.items():
                    ni = int(ni)
                    predicate_heights = predicates['heights']
                    predicate_shortest_path_mat = predicates['shortest_path_mat']
                    predicate_num = len(predicate_heights)
                    
                    predicate_dist = torch.tensor(predicate_shortest_path_mat, dtype=torch.int)
                    max_dist = min(predicate_num, self.max_attn_dist)
                    
                    for ii in range(index_num):
                        predicate_tree_idx = batch_feat[f'{pt}_predicate_tree_idx_mapping'][si, ii, ni]
                        
                        batch_feat[f'{pt}_heights'][predicate_tree_idx, :predicate_num] = torch.tensor(predicate_heights, dtype=torch.int)
                        batch_feat[f'{pt}_rel_pos_mat'][predicate_tree_idx, 1:predicate_num+1, 1:predicate_num+1] = predicate_dist
                        batch_feat[f'{pt}_tree_attn_bias'][predicate_tree_idx, 1:predicate_num+1, 1:predicate_num+1][predicate_dist >= max_dist] = float('-inf')
                        batch_feat[f'{pt}_tree_attn_bias'][predicate_tree_idx, :, predicate_num+1:] = float('-inf')
                
            if self.log_label:
                batch_label[si] = math.log(1 + sample['label'])
            else:
                batch_label[si] = sample['label']

        return (batch_feat, batch_label)
