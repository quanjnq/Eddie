"""
This code is based on the implementation from the original repository of [QueryFormer].
Source code: https://github.com/zhaoyue-ntu/QueryFormer/blob/main/model/database_util.py

For detailed changes, look for comments marked as 'MODIFIED' in the code.
"""
import numpy as np
import pandas as pd
import csv
import torch

## bfs shld be enough
def floyd_warshall_rewrite(adjacency_matrix, max_dist=160): # # MODIFIED: Added max_dist parameter to replace hardcoded value
    (nrows, ncols) = adjacency_matrix.shape
    assert nrows == ncols
    M = adjacency_matrix.copy().astype('long')
    for i in range(nrows):
        for j in range(ncols):
            if i == j: 
                M[i][j] = 0
            elif M[i][j] == 0: 
                M[i][j] = max_dist+1
    
    for k in range(nrows):
        for i in range(nrows):
            for j in range(nrows):
                M[i][j] = min(M[i][j], M[i][k]+M[k][j])
    return M

def get_job_table_sample(workload_file_name, num_materialized_samples = 1000):

    tables = []
    samples = []

    # Load queries
    with open(workload_file_name + ".csv", 'r') as f:
        data_raw = list(list(rec) for rec in csv.reader(f, delimiter='#'))
        for row in data_raw:
            tables.append(row[0].split(','))

            if int(row[3]) < 1:
                print("Queries must have non-zero cardinalities")
                exit(1)

    print("Loaded queries with len ", len(tables))
    
    # Load bitmaps
    num_bytes_per_bitmap = int((num_materialized_samples + 7) >> 3)
    with open(workload_file_name + ".bitmaps", 'rb') as f:
        for i in range(len(tables)):
            four_bytes = f.read(4)
            if not four_bytes:
                print("Error while reading 'four_bytes'")
                exit(1)
            num_bitmaps_curr_query = int.from_bytes(four_bytes, byteorder='little')
            bitmaps = np.empty((num_bitmaps_curr_query, num_bytes_per_bitmap * 8), dtype=np.uint8)
            for j in range(num_bitmaps_curr_query):
                # Read bitmap
                bitmap_bytes = f.read(num_bytes_per_bitmap)
                if not bitmap_bytes:
                    print("Error while reading 'bitmap_bytes'")
                    exit(1)
                bitmaps[j] = np.unpackbits(np.frombuffer(bitmap_bytes, dtype=np.uint8))
            samples.append(bitmaps)
    print("Loaded bitmaps")
    table_sample = []
    for ts, ss in zip(tables,samples):
        d = {}
        for t, s in zip(ts,ss):
            tf = t.split(' ')[0] # remove alias
            d[tf] = s
        table_sample.append(d)
    
    return table_sample


def get_hist_file(hist_path, bin_number = 50):
    hist_file = pd.read_csv(hist_path)
    
    # MODIFIED: Removed handling of 'freq' key as it is not present in hist_file.
    # for i in range(len(hist_file)):
    #     freq = hist_file['freq'][i]
    #     freq_np = np.frombuffer(bytes.fromhex(freq), dtype=float)
    #     hist_file['freq'][i] = freq_np

    # MODIFIED: Removed code for regenerating 'table_column' in hist_file to prevent errors in filterDict2Hist().
    # table_column = []
    # for i in range(len(hist_file)):
    #     table = hist_file['table'][i]
    #     col = hist_file['column'][i]
    #     table_alias = ''.join([tok[0] for tok in table.split('_')])
    #     if table == 'movie_info_idx': table_alias = 'mi_idx'
    #     combine = '.'.join([table_alias,col])
    #     table_column.append(combine)
    # hist_file['table_column'] = table_column
    
    for rid in range(len(hist_file)):
        # MODIFIED: Convert bin values to int using float conversion to handle cases like "0.0" and avoid errors.
        hist_file['bins'][rid] = \
            [int(float(i)) for i in hist_file['bins'][rid][1:-1].split(' ') if len(i)>0] 

    if bin_number != 50:
        hist_file = re_bin(hist_file, bin_number)

    return hist_file

def re_bin(hist_file, target_number):
    for i in range(len(hist_file)):
        freq = hist_file['freq'][i]
        bins = freq2bin(freq,target_number)
        hist_file['bins'][i] = bins
    return hist_file

def freq2bin(freqs, target_number):
    freq = freqs.copy()
    maxi = len(freq)-1
    
    step = 1. / target_number
    mini = 0
    while freq[mini+1]==0:
        mini+=1
    pointer = mini+1
    cur_sum = 0
    res_pos = [mini]
    residue = 0
    while pointer < maxi+1:
        cur_sum += freq[pointer]
        freq[pointer] = 0
        if cur_sum >= step:
            cur_sum -= step
            res_pos.append(pointer)
        else:
            pointer += 1
    
    if len(res_pos)==target_number: res_pos.append(maxi)
    
    return res_pos



class Batch():
    def __init__(self, attn_bias, rel_pos, heights, x, y=None):
        super(Batch, self).__init__()

        self.heights = heights
        self.x, self.y = x, y
        self.attn_bias = attn_bias
        self.rel_pos = rel_pos
        
    def to(self, device):

        self.heights = self.heights.to(device)
        self.x = self.x.to(device)

        self.attn_bias, self.rel_pos = self.attn_bias.to(device), self.rel_pos.to(device)

        return self

    def __len__(self):
        return self.in_degree.size(0)


def pad_1d_unsqueeze(x, padlen):
    x = x + 1 # pad id = 0
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen], dtype=x.dtype)
        new_x[:xlen] = x
        x = new_x
    return x.unsqueeze(0)


def pad_2d_unsqueeze(x, padlen):
    # dont know why add 1, comment out first
#    x = x + 1 # pad id = 0
    xlen, xdim = x.size()
    if xlen < padlen:
        new_x = x.new_zeros([padlen, xdim], dtype=x.dtype) + 1
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


def collator(batch):
    # MODIFIED: Changed input and output format to integrate with a unified process. 
    feats = [sample[0] for sample in batch]
    y = torch.tensor([sample[1] for sample in batch], dtype=torch.float)
    
    feat_dict = {
        'plan_x': torch.cat([s['plan_feat']['x'] for s in feats]),
        'plan_attn_bias': torch.cat([s['plan_feat']['attn_bias'] for s in feats]), 
        'plan_rel_pos': torch.cat([s['plan_feat']['rel_pos'] for s in feats]), 
        'plan_heights': torch.cat([s['plan_feat']['heights'] for s in feats]), 
        
        'index_feat': torch.cat([s['index_feat'] for s in feats])
    }
    if 'plan_hypo_feat' in feats[0]:
        hypo_feat_dict = {'plan_hypo_x': torch.cat([s['plan_hypo_feat']['x'] for s in feats]),
        'plan_hypo_attn_bias': torch.cat([s['plan_hypo_feat']['attn_bias'] for s in feats]), 
        'plan_hypo_rel_pos': torch.cat([s['plan_hypo_feat']['rel_pos'] for s in feats]), 
        'plan_hypo_heights': torch.cat([s['plan_hypo_feat']['heights'] for s in feats]), 
        }
        feat_dict.update(hypo_feat_dict)
    
    return (feat_dict, 
            y)


def filterDict2Hist(hist_file, filterDict, encoding):
    buckets = len(hist_file['bins'][0]) 
    empty = np.zeros(buckets - 1)
    ress = np.zeros((3, buckets-1))
    for i in range(len(filterDict['colId'])):
        colId = filterDict['colId'][i]
        col = encoding.idx2col[colId]
        if col == 'NA':
            ress[i] = empty
            continue
        bins = hist_file.loc[hist_file['table_column']==col,'bins'].item()
        
        opId = filterDict['opId'][0]
        op = encoding.idx2op[opId]
        
        val = filterDict['val'][0]
        mini, maxi = encoding.column_min_max_vals[col]
        val_unnorm = val * (maxi-mini) + mini
        
        left = 0
        right = len(bins)-1
        for j in range(len(bins)):
            if bins[j]<val_unnorm:
                left = j
            if bins[j]>val_unnorm:
                right = j
                break

        res = np.zeros(len(bins)-1)

        if op == '=':
            res[left:right] = 1
        elif op == '<':
            res[:left] = 1
        elif op == '>':
            res[right:] = 1
        ress[i] = res
    
    ress = ress.flatten()
    return ress     
        



def formatJoin(json_node):
   
    join = None
    if 'Hash Cond' in json_node:
        join = json_node['Hash Cond']
    elif 'Join Filter' in json_node:
        join = json_node['Join Filter']
    ## TODO: index cond
    elif 'Index Cond' in json_node and not json_node['Index Cond'][-2].isnumeric():
        join = json_node['Index Cond']
    
    ## sometimes no alias, say t.id 
    ## remove repeat (both way are the same)
    if join is not None:

        twoCol = join[1:-1].split(' = ')
        # MODIFIED: handle Alias not in json_node
        if "Alias"  not in json_node and min([len(col.split('.')) for col in twoCol]) == 1:
            return None
        twoCol = [json_node['Alias'] + '.' + col 
                  if len(col.split('.')) == 1 else col for col in twoCol ] 
        join = ' = '.join(sorted(twoCol))
    
    return join
    
def formatFilter(plan):
    alias = None
    if 'Alias' in plan:
        alias = plan['Alias']
    else:
        pl = plan
        while 'parent' in pl:
            pl = pl['parent']
            if 'Alias' in pl:
                alias = pl['Alias']
                break
    
    filters = []
    if 'Filter' in plan:
        filters.append(plan['Filter'])
    if 'Index Cond' in plan and plan['Index Cond'][-2].isnumeric():
        filters.append(plan['Index Cond'])
    if 'Recheck Cond' in plan:
        filters.append(plan['Recheck Cond'])
        
    
    
    return filters, alias

class Encoding:
    def __init__(self, column_min_max_vals, 
                 col2idx, op2idx={'>':0, '=':1, '<':2, 'NA':3}):
        self.op2idx = op2idx
        
        idx2col = {}
        for k,v in col2idx.items():
            idx2col[v] = k
        self.idx2col = idx2col
        self.idx2op = {0:'>', 1:'=', 2:'<', 3:'NA'}
        
        # MODIFIED: Handle cases where column name does not have table name as prefix.
        tbl_cols = [str(tbl_col) for tbl_col in col2idx.keys() if tbl_col != 'NA']
        for tbl_col in tbl_cols:
            col = tbl_col.split('.')[-1]
            column_min_max_vals[col] = column_min_max_vals[tbl_col]
            col2idx[col] = col2idx[tbl_col]
        
        self.column_min_max_vals = column_min_max_vals
        self.col2idx = col2idx
        
        self.type2idx = {}
        self.idx2type = {}
        self.join2idx = {}
        self.idx2join = {}
        
        self.table2idx = {'NA':0}
        self.idx2table = {0:'NA'}
    
    def normalize_val(self, column, val, log=False):
        mini, maxi = self.column_min_max_vals[column]
        
        val_norm = 0.0
        if maxi > mini:
            val_norm = (val-mini) / (maxi-mini)
        return val_norm
    
    def encode_filters(self, filters=[], alias=None): 
        ## filters: list of dict 

#        print(filt, alias)
        if len(filters) == 0:
            return {'colId':[self.col2idx['NA']],
                   'opId': [self.op2idx['NA']],
                   'val': [0.0]} 
        res = {'colId':[],'opId': [],'val': []}
        for filt in filters:
            filt = ''.join(c for c in filt if c not in '()')
            fs = filt.split(' AND ')
            for f in fs:
     #           print(filters)
                col, op, num = f.split(' ')[:3] # MODIFIED: Truncate to 3 elements to avoid errors.
                if alias is not None: # MODIFIED: Added a check for alias to avoid errors when alias is None.
                    column = alias + '.' + col
                else:
                    column = col
    #            print(f)
                
                # MODIFIED: Assign default values for unseen col/op.
                res['colId'].append(self.col2idx[column] if column in self.col2idx else self.col2idx['NA'])
                res['opId'].append(self.op2idx[op] if op in self.op2idx else self.op2idx['NA'])
                try:
                    res['val'].append(self.normalize_val(column, float(num)))
                except:
                    res['val'].append(0)
        return res
    
    def encode_join(self, join):
        if join not in self.join2idx:
            self.join2idx[join] = len(self.join2idx)
            self.idx2join[self.join2idx[join]] = join
        return self.join2idx[join]
    
    def encode_table(self, table):
        if table not in self.table2idx:
            self.table2idx[table] = len(self.table2idx)
            self.idx2table[self.table2idx[table]] = table
        return self.table2idx[table]

    def encode_type(self, nodeType):
        if nodeType not in self.type2idx:
            self.type2idx[nodeType] = len(self.type2idx)
            self.idx2type[self.type2idx[nodeType]] = nodeType
        return self.type2idx[nodeType]


class TreeNode:
    def __init__(self, nodeType, typeId, filt, card, join, join_str, filterDict):
        self.nodeType = nodeType
        self.typeId = typeId
        self.filter = filt
        
        self.table = 'NA'
        self.table_id = 0
        self.query_id = None ## so that sample bitmap can recognise
        
        self.join = join
        self.join_str = join_str
        self.card = card #'Actual Rows'
        self.children = []
        self.rounds = 0
        
        self.filterDict = filterDict
        
        self.parent = None
        
        self.feature = None
        
    def addChild(self,treeNode):
        self.children.append(treeNode)
    
    def __str__(self):
#        return TreeNode.print_nested(self)
        return '{} with {}, {}, {} children'.format(self.nodeType, self.filter, self.join_str, len(self.children))

    def __repr__(self):
        return self.__str__()
    
    @staticmethod
    def print_nested(node, indent = 0): 
        print('--'*indent+ '{} with {} and {}, {} childs'.format(node.nodeType, node.filter, node.join_str, len(node.children)))
        for k in node.children: 
            TreeNode.print_nested(k, indent+1)
        





