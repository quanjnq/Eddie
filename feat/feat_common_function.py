import numpy as np
import torch
import logging
import sys
import re


COL_NAME_MODE = re.compile(r'([a-zA-Z_][a-zA-Z0-9_.]*)')

def floyd_warshall_rewrite(adjacency_matrix):
    (nrows, ncols) = adjacency_matrix.shape
    assert nrows == ncols
    M = adjacency_matrix.copy().astype('long')
    for i in range(nrows):
        for j in range(ncols):
            if i == j:
                M[i][j] = 0
            elif M[i][j] == 0:
                M[i][j] = nrows

    for k in range(nrows):
        for i in range(nrows):
            for j in range(nrows):
                M[i][j] = min(M[i][j], M[i][k] + M[k][j])
    return M


def calculate_height(adj_list, tree_size):
    if tree_size == 1:
        return np.array([0])

    adj_list = np.array(adj_list)
    node_ids = np.arange(tree_size, dtype=int)
    node_order = np.zeros(tree_size, dtype=int)
    uneval_nodes = np.ones(tree_size, dtype=bool)

    parent_nodes = adj_list[:, 0]
    child_nodes = adj_list[:, 1]

    n = 0
    while uneval_nodes.any():
        uneval_mask = uneval_nodes[child_nodes]
        unready_parents = parent_nodes[uneval_mask]

        node2eval = uneval_nodes & ~np.isin(node_ids, unready_parents)
        node_order[node2eval] = n
        uneval_nodes[node2eval] = False
        n += 1
    return node_order


def calculate_struct(adj_list, node_num):
    heights = calculate_height(adj_list, node_num)

    edge_index = torch.LongTensor(np.array(adj_list)).t()
    if len(edge_index) == 0:
        shortest_path_mat = np.array([[0]])
    else:
        adj = torch.zeros([node_num, node_num], dtype=torch.bool)
        adj[edge_index[0, :], edge_index[1, :]] = True

        shortest_path_mat = floyd_warshall_rewrite(adj.numpy())
    return heights, shortest_path_mat


def get_full_col_name(tbl_col, tbl_name_of_col, alias2relation=None):
    tbl_name_of_col = tbl_name_of_col if tbl_name_of_col is not None else "NULL"
    tbl_col = f"{tbl_name_of_col}.{tbl_col}" if '.' not in tbl_col else tbl_col
    sp = tbl_col.split(".")
    relation = sp[0]
    if relation.startswith('"') and relation.endswith('"'):
        relation = relation[1:-1]
    col = sp[1]
    if alias2relation is not None and relation in alias2relation:
        relation = alias2relation[relation]
    return relation + '.' + col

def pad_1d_unsqueeze(x, padlen):
    x = x + 1 # pad id = 0
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen], dtype=x.dtype)
        new_x[:xlen] = x
        x = new_x
    return x.unsqueeze(0)


def pad_2d_unsqueeze(x, padlen):
    xlen, xdim = x.size()
    if xlen < padlen:
        new_x = x.new_zeros([padlen, xdim], dtype=x.dtype)
        new_x[:xlen, :] = x
        x = new_x
    return x.unsqueeze(0)


def pad_rel_pos_unsqueeze(x, padlen):
    x = x + 1
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen, padlen], dtype=x.dtype)
        new_x[:xlen, :xlen] = x
        x = new_x
    return x.unsqueeze(0)


def pad_attn_bias_unsqueeze(x, padlen):
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen, padlen], dtype=x.dtype).fill_(float('-inf'))
        new_x[:xlen, :xlen] = x
        new_x[xlen:, :xlen] = 0
        x = new_x
    return x.unsqueeze(0)


def is_col_name(colname):
    return re.match(COL_NAME_MODE, colname) is not None


def encode_col_feat(raw_tbl_col, db_stats, index_tbl_name, alias2relation, index_col2order, tbl_name_of_col=None):
    if alias2relation is not None and len(alias2relation) == 1 and tbl_name_of_col is None:
        for k, v in alias2relation.items():
            tbl_name_of_col = v

    full_col_name = get_full_col_name(raw_tbl_col, tbl_name_of_col, alias2relation)
    if full_col_name in db_stats:
        col_stat = db_stats[full_col_name]
    else:
        logging.debug(f"unkown col, raw_tbl_col={raw_tbl_col} full_col_name={full_col_name}")
        col_stat = None
    tbl_name = full_col_name.split(".")[0]
    col_name = full_col_name.split(".")[1]
    if tbl_name == index_tbl_name and col_name in index_col2order:
        index_pos = index_col2order[col_name]
    elif tbl_name != index_tbl_name and full_col_name in db_stats:
        index_pos =  index_col2order["NOT_SAME_TBL"]
    else:
        index_pos =  index_col2order["UNK"]
        
    col_feat = {
        "col_name": full_col_name,
        "raw_col_name": raw_tbl_col,
        "index_pos": index_pos,
        "data_type": col_stat['data_type'] if col_stat else None, 
        "dist_cnt": col_stat['n_distinct'] if col_stat else None, 
        "dist_frac": col_stat['n_distinct'] * 1.0 / col_stat['rows'] if col_stat else None, 
        "null_frac": col_stat['null_frac'] if col_stat else None
    }
    return col_feat