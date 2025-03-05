"""
This code is based on the implementation from the original repository of [QueryFormer].
Source code: https://github.com/zhaoyue-ntu/QueryFormer/blob/main/model/dataset.py

For detailed changes, look for comments marked as 'MODIFIED' in the code.
"""

import torch
from torch.utils.data import Dataset
import numpy as np
import json
import pandas as pd
import sys, os
from collections import deque
from .database_util import formatFilter, formatJoin, TreeNode, filterDict2Hist
from .database_util import *
import logging


class QfDataset(Dataset):
    def __init__(self, data, encoding, hist_file, table_sample, is_aimai=False):
        # MODIFIED: Changed Dataset input and output format to integrate with a unified process. 
        
        self.table_sample = table_sample
        self.encoding = encoding
        self.hist_file = hist_file
        self.is_aimai = is_aimai
        
        self.length = len(data)
        
        nodes = [sample["plan_tree"] for sample in data]
        
        nodes_hypo = [sample["plan_hypo"] for sample in data] if is_aimai else [None for i in range(len(data))]
        self.labels = [sample["label"] for sample in data]
        
        index_feats = []
        for indexes in [sample["indexes"] for sample in data]:
            index_feat = [encoding.col2idx[col] if col in encoding.col2idx else encoding.col2idx['NA'] for index in indexes for col in index]
            index_feats.append(index_feat)
        
        idxs = list(range(self.length))
    
        self.treeNodes = [] ## for mem collection
        self.collated_dicts = [self.js_node2dict(i,node,node_hypo,index_feat) for i,node,node_hypo,index_feat in zip(idxs, nodes, nodes_hypo, index_feats)]

        
    def js_node2dict(self, idx, node, nodes_hypo, index_feat):
        # MODIFIED: add features of plans for hypothetical indexes and index features. 
        
        if idx % 100 == 0:
            logging.info(f"{idx} data items processed")
            
        treeNode = self.traversePlan(node, idx, self.encoding)
        if self.is_aimai:
            treeNode_hypo = self.traversePlan(nodes_hypo, idx, self.encoding)
        
        _dict = {}
        _dict['plan_feat'] = self.node2dict(treeNode)
        if self.is_aimai:
            _dict['plan_hypo_feat'] = self.node2dict(treeNode_hypo)
        _dict['index_feat'] = torch.LongTensor(index_feat)
        collated_dict = self.pre_collate(_dict)
        
        self.treeNodes.clear()
        del self.treeNodes[:]

        return collated_dict

    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        return (self.collated_dicts[idx], self.labels[idx])
    
    ## pre-process first half of old collator
    def pre_collate(self, the_dict, max_node = 160, rel_pos_max = 160, max_index_num=5, max_index_col=3):
        # MODIFIED: add features of plans for hypothetical indexes and index features. 
        
        collated = {}
        plan_type_list = ['plan_feat', 'plan_hypo_feat'] if self.is_aimai else ['plan_feat']
        
        for plan_type in plan_type_list:
            plan_dict = the_dict[plan_type]
            
            x = pad_2d_unsqueeze(plan_dict['features'], max_node)
            N = len(plan_dict['features'])
            attn_bias = torch.zeros([N+1,N+1], dtype=torch.float)
            
            edge_index = plan_dict['adjacency_list'].t()
            if len(edge_index) == 0:
                shortest_path_result = np.array([[0]])
                path = np.array([[0]])
                adj = torch.tensor([[0]]).bool()
            else:
                adj = torch.zeros([N,N], dtype=torch.bool)
                adj[edge_index[0,:], edge_index[1,:]] = True
                
                shortest_path_result = floyd_warshall_rewrite(adj.numpy())
            
            rel_pos = torch.from_numpy((shortest_path_result)).long()

            
            attn_bias[1:, 1:][rel_pos >= rel_pos_max] = float('-inf')
            
            attn_bias = pad_attn_bias_unsqueeze(attn_bias, max_node + 1)
            rel_pos = pad_rel_pos_unsqueeze(rel_pos, max_node)

            heights = pad_1d_unsqueeze(plan_dict['heights'], max_node)
            
            collated[plan_type] = {
                'x' : x,
                'attn_bias': attn_bias,
                'rel_pos': rel_pos,
                'heights': heights,
            }
        
        collated['index_feat'] = pad_1d_unsqueeze(the_dict['index_feat'], max_index_num*max_index_col) # MODIFIED: add index features
        
        return collated


    def node2dict(self, treeNode):

        adj_list, num_child, features = self.topo_sort(treeNode)
        heights = self.calculate_height(adj_list, len(features))

        return {
            'features' : torch.FloatTensor(np.array(features)),
            'heights' : torch.LongTensor(heights),
            'adjacency_list' : torch.LongTensor(np.array(adj_list)),
          
        }
    
    def topo_sort(self, root_node):
#        nodes = []
        adj_list = [] #from parent to children
        num_child = []
        features = []

        toVisit = deque()
        toVisit.append((0,root_node))
        next_id = 1
        while toVisit:
            idx, node = toVisit.popleft()
#            nodes.append(node)
            features.append(node.feature)
            num_child.append(len(node.children))
            for child in node.children:
                toVisit.append((next_id,child))
                adj_list.append((idx,next_id))
                next_id += 1
        
        return adj_list, num_child, features
    
    def traversePlan(self, plan, idx, encoding): # bfs accumulate plan
        if not plan:
            return

        nodeType = plan['Node Type']
        typeId = encoding.encode_type(nodeType)
        card = None #plan['Actual Rows']
        filters, alias = formatFilter(plan)
        join = formatJoin(plan)
        joinId = encoding.encode_join(join)
        filters_encoded = encoding.encode_filters(filters, alias)
        
        root = TreeNode(nodeType, typeId, filters, card, joinId, join, filters_encoded)
        
        self.treeNodes.append(root)

        if 'Relation Name' in plan:
            root.table = plan['Relation Name']
            root.table_id = encoding.encode_table(plan['Relation Name'])
        root.query_id = idx
        
        root.feature = node2feature(root, encoding, self.hist_file, self.table_sample)
        #    print(root)
        if 'Plans' in plan:
            for subplan in plan['Plans']:
                subplan['parent'] = plan
                node = self.traversePlan(subplan, idx, encoding)
                node.parent = root
                root.addChild(node)
        return root

    def calculate_height(self, adj_list,tree_size):
        if tree_size == 1:
            return np.array([0])

        adj_list = np.array(adj_list)
        node_ids = np.arange(tree_size, dtype=int)
        node_order = np.zeros(tree_size, dtype=int)
        uneval_nodes = np.ones(tree_size, dtype=bool)

        parent_nodes = adj_list[:,0]
        child_nodes = adj_list[:,1]

        n = 0
        while uneval_nodes.any():
            uneval_mask = uneval_nodes[child_nodes]
            unready_parents = parent_nodes[uneval_mask]

            node2eval = uneval_nodes & ~np.isin(node_ids, unready_parents)
            node_order[node2eval] = n
            uneval_nodes[node2eval] = False
            n += 1
        return node_order 



def node2feature(node, encoding, hist_file, table_sample):
    # type, join, filter123, mask123
    # 1, 1, 3x3 (9), 3
    # TODO: add sample (or so-called table)
    
    # MODIFIED: Truncate to max predicate num (3) to prevent errors
    node.filterDict['colId'] = node.filterDict['colId'][:3]
    node.filterDict['opId'] = node.filterDict['opId'][:3]
    node.filterDict['val'] = node.filterDict['val'][:3]
    
    num_filter = len(node.filterDict['colId'])
    pad = np.zeros((3,3-num_filter))
    filts = np.array(list(node.filterDict.values())) #cols, ops, vals
    ## 3x3 -> 9, get back with reshape 3,3
    filts = np.concatenate((filts, pad), axis=1).flatten() 
    mask = np.zeros(3)
    mask[:num_filter] = 1
    type_join = np.array([node.typeId, node.join])
    
    hists = filterDict2Hist(hist_file, node.filterDict, encoding)


    # table, bitmap, 1 + 1000 bits
    table = np.array([node.table_id])
    if table_sample is None or node.table_id == 0: # MODIFIED: Handle cases where table_sample is None.
        sample = np.zeros(1000)
    else:
        sample = table_sample[node.query_id][node.table]

    #return np.concatenate((type_join,filts,mask))
    return np.concatenate((type_join, filts, mask, hists, table, sample))

