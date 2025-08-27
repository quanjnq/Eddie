import torch
import torch.nn as nn
import torch.nn.functional as F


class Prediction(nn.Module):
    def __init__(self, in_feature = 128, hid_units = 128, contract = 1, mid_layers = True, res_con = True, clip_label=True):
        super(Prediction, self).__init__()
        self.mid_layers = mid_layers
        self.res_con = res_con
        
        self.out_mlp1 = nn.Linear(in_feature, hid_units)

        self.mid_mlp1 = nn.Linear(hid_units, hid_units//contract)
        self.mid_mlp2 = nn.Linear(hid_units//contract, hid_units)

        self.out_mlp2 = nn.Linear(hid_units, 1)
        
        self.clip_label = clip_label

    def forward(self, features):
        
        hid = F.relu(self.out_mlp1(features))
        if self.mid_layers:
            mid = F.relu(self.mid_mlp1(hid))
            mid = F.relu(self.mid_mlp2(mid))
            if self.res_con:
                hid = hid + mid
            else:
                hid = mid
        out = torch.sigmoid(self.out_mlp2(hid)) if self.clip_label else self.out_mlp2(hid)

        return out


class FeedForwardNetwork(nn.Module):
    def __init__(self, hidden_size, ffn_size, dropout_rate):
        super(FeedForwardNetwork, self).__init__()

        self.layer1 = nn.Linear(hidden_size, ffn_size)
        self.gelu = nn.GELU()
        self.layer2 = nn.Linear(ffn_size, hidden_size)

    def forward(self, x):
        x = self.layer1(x)
        x = self.gelu(x)
        x = self.layer2(x)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, attention_dropout_rate, head_size, att_size=None):
        super(MultiHeadAttention, self).__init__()

        self.head_size = head_size

        if att_size:
            self.att_size = att_size
        else:
            self.att_size = att_size = hidden_size // head_size
        self.scale = att_size ** -0.5

        self.linear_q = nn.Linear(hidden_size, head_size * att_size)
        self.linear_k = nn.Linear(hidden_size, head_size * att_size)
        self.linear_v = nn.Linear(hidden_size, head_size * att_size)
        self.att_dropout = nn.Dropout(attention_dropout_rate)

        self.output_layer = nn.Linear(head_size * att_size, head_size * att_size)

    def forward(self, q, k, v, attn_bias=None, mask=None, enable_out_layer=True):
        orig_q_size = q.size()

        d_k = self.att_size
        d_v = self.att_size
        batch_size = q.size(0)

        # head_i = Attention(Q(W^Q)_i, K(W^K)_i, V(W^V)_i)
        q = self.linear_q(q).reshape(batch_size, -1, self.head_size, d_k)
        k = self.linear_k(k).reshape(batch_size, -1, self.head_size, d_k)
        v = self.linear_v(v).reshape(batch_size, -1, self.head_size, d_v)

        q = q.transpose(1, 2)                  # [b, h, q_len, d_k]
        v = v.transpose(1, 2)                  # [b, h, v_len, d_v]
        k = k.transpose(1, 2).transpose(2, 3)  # [b, h, d_k, k_len]

        # Scaled Dot-Product Attention.
        # Attention(Q, K, V) = softmax((QK^T)/sqrt(d_k))V
        q = q * self.scale
        x = torch.matmul(q, k)  # [b, h, q_len, k_len]
        
        if mask is not None:
            k_mask = mask.unsqueeze(1).unsqueeze(2) # [b, 1, 1, k_len]
            x = x.masked_fill(k_mask == 0, -1e9)
        
        if attn_bias is not None:
            x = x + attn_bias

        x = torch.softmax(x, dim=3)
        x = self.att_dropout(x)
        x = x.matmul(v)  # [b, h, q_len, attn]
        
        if mask is not None:
            x_mask = mask.unsqueeze(1).unsqueeze(3) # [b, 1, q_len, 1]
            x *= x_mask
            
        x = x.transpose(1, 2).contiguous()  # [b, q_len, h, attn]
        x = x.reshape(batch_size, -1, self.head_size * d_v)

        if enable_out_layer:
            x = self.output_layer(x)

        # assert x.size() == orig_q_size
        return x
    
    
class EncoderLayer(nn.Module):
    def __init__(self, hidden_size=128, ffn_size=32, dropout_rate=0.1, attention_dropout_rate=0.1, head_size=4, att_size=None):
        super(EncoderLayer, self).__init__()

        self.self_attention_norm = nn.LayerNorm(hidden_size)
        self.self_attention = MultiHeadAttention(hidden_size, attention_dropout_rate, head_size, att_size)
        self.self_attention_dropout = nn.Dropout(dropout_rate)

        self.ffn_norm = nn.LayerNorm(hidden_size)
        self.ffn = FeedForwardNetwork(hidden_size, ffn_size, dropout_rate)
        self.ffn_dropout = nn.Dropout(dropout_rate)

    def forward(self, x, attn_bias=None, mask=None):
        y = self.self_attention_norm(x)
        y = self.self_attention(y, y, y, attn_bias, mask) 
        y = self.self_attention_dropout(y)
        x = x + y

        y = self.ffn_norm(x)
        y = self.ffn(y)
        y = self.ffn_dropout(y)
        x = x + y
        return x


class EncoderLayerSimple(nn.Module):
    def __init__(self, hidden_size=128, ffn_size=32, dropout_rate=0.1, attention_dropout_rate=0.1, head_size=4, att_size=None):
        super(EncoderLayerSimple, self).__init__()

        self.self_attention_norm = nn.LayerNorm(hidden_size)
        self.self_attention = MultiHeadAttention(hidden_size, attention_dropout_rate, head_size, att_size)
        self.self_attention_dropout = nn.Dropout(dropout_rate)

    def forward(self, x, attn_bias=None, mask=None):
        y = self.self_attention_norm(x)
        y = self.self_attention(y, y, y, attn_bias, mask, enable_out_layer=False) 
        y = self.self_attention_dropout(y)
        return y
    
    
class TreeBiasAttention(nn.Module):
    def __init__(self, ffn_dim=32, head_size=4, \
                 dropout=0.1, attention_dropout_rate=0.1, n_layers=4, \
                 hidden_dim=128
                 ):

        super(TreeBiasAttention, self).__init__()
        
        self.super_node_embedding = nn.Embedding(1, hidden_dim)
        self.head_size = head_size
        self.rel_pos_embedding = nn.Embedding(192, head_size, padding_idx=0) # Modified: 128 -> 192
        self.super_node_virtual_distance = nn.Embedding(1, head_size)
        self.input_dropout = nn.Dropout(dropout)
        
        encoders = [EncoderLayer(hidden_dim, ffn_dim, dropout, attention_dropout_rate, head_size)
                    for _ in range(n_layers)]
        self.layers = nn.ModuleList(encoders)

        self.final_ln = nn.LayerNorm(hidden_dim)

    def forward(self, node_embedded, rel_pos_mat, tree_attn_bias):
        n_batch = node_embedded.size()[0]
        
        super_node_embedded = self.super_node_embedding.weight.unsqueeze(0).repeat(n_batch, 1, 1)
        all_node_embedded = torch.cat([super_node_embedded, node_embedded], dim=1)  # [n_batch, n_node+1, hidden_dim]
        
        # tree attention bias
        tree_attn_bias = tree_attn_bias.unsqueeze(1).repeat(1, self.head_size, 1, 1)    # [n_batch, n_head, n_node+1, n_node+1]
        rel_pos_bias = self.rel_pos_embedding(rel_pos_mat).permute(0, 3, 1, 2)  # [n_batch, n_node+1, n_node+1, n_head] -> [n_batch, n_head, n_node+1, n_node+1]
        tree_attn_bias = tree_attn_bias + rel_pos_bias
        # tree_attn_bias = tree_attn_bias
        
        # reset tree attention bias for super node
        t = self.super_node_virtual_distance.weight.reshape(1, self.head_size, 1)
        tree_attn_bias[:, :, 1:, 0] = tree_attn_bias[:, :, 1:, 0] + t
        tree_attn_bias[:, :, 0, :] = tree_attn_bias[:, :, 0, :] + t
        
        # transfomrer encoder
        output = self.input_dropout(all_node_embedded)
        for enc_layer in self.layers:
            output = enc_layer(output, tree_attn_bias)
        output = self.final_ln(output) # [n_batch, n_node+1, hidden_dim]
        return output
        


class NodeIndexAttention(nn.Module):
    def __init__(self, ffn_dim=32, head_size=4, \
                 dropout=0.1, attention_dropout_rate=0.1, n_layers=4, \
                 hidden_dim=128, disable_idx_attn=False
                 ):

        super(NodeIndexAttention, self).__init__()
        self.super_node_embedding = nn.Embedding(1, hidden_dim)
        self.head_size = head_size
        self.hidden_dim = hidden_dim
        self.rel_pos_embedding = nn.Embedding(192, head_size, padding_idx=0) # Modifid: 128 -> 192
        self.super_node_virtual_distance = nn.Embedding(1, head_size)
        self.input_dropout = nn.Dropout(dropout)
        self.disable_idx_attn = disable_idx_attn
        
        self.encoder_layers1 = nn.ModuleList()
        self.encoder_layers2 = nn.ModuleList()
        self.comb_layers = nn.ModuleList()
        for _ in range(n_layers):
            self.encoder_layers1.append(EncoderLayerSimple(hidden_dim, ffn_dim, dropout, attention_dropout_rate, head_size, att_size=hidden_dim // head_size // 2))
            self.encoder_layers2.append(EncoderLayerSimple(hidden_dim, ffn_dim, dropout, attention_dropout_rate, head_size, att_size=hidden_dim // head_size // 2))
            self.comb_layers.append(nn.Linear(hidden_dim*2, hidden_dim))

        self.ffn_norm = nn.LayerNorm(hidden_dim)
        self.ffn = FeedForwardNetwork(hidden_dim, ffn_dim, attention_dropout_rate)
        self.ffn_dropout = nn.Dropout(attention_dropout_rate)
        self.attn_output_layer = nn.Linear(hidden_dim, hidden_dim)

        self.final_ln = nn.LayerNorm(hidden_dim)

    def forward(self, node_embedded, heights, rel_pos_mat, tree_attn_bias, index_attn_bias):
        rel_pos_mat = rel_pos_mat.reshape(-1, *rel_pos_mat.shape[-2:]) # [n_batch * n_index, n_node, n_node]
        tree_attn_bias = tree_attn_bias.reshape(-1, *tree_attn_bias.shape[-2:]) # [n_batch * n_index, n_node, n_node]

        n_batch = node_embedded.size()[0]
        n_index = node_embedded.size()[1]
        
        node_embedded = node_embedded + heights  # [n_batch, n_index, n_node, hidden_dim]

        super_node_embedded = self.super_node_embedding.weight.unsqueeze(0).repeat(n_batch, n_index, 1, 1)
        all_node_embedded = torch.cat([super_node_embedded, node_embedded], dim=2)  # [n_batch, n_index, n_node+1, hidden_dim]

        # tree attention bias
        tree_attn_bias = tree_attn_bias.unsqueeze(1).repeat(1, self.head_size, 1, 1)    # [n_batch * n_index, n_head, n_node+1, n_node+1]
        rel_pos_bias = self.rel_pos_embedding(rel_pos_mat).permute(0, 3, 1, 2)  # [n_batch * n_index, n_node+1, n_node+1, n_head] -> [n_batch * n_index, n_head, n_node+1, n_node+1]
        tree_attn_bias = tree_attn_bias + rel_pos_bias # [n_batch * n_index, n_head, n_node+1, n_node+1]
        
        # reset tree attention bias for super node
        t = self.super_node_virtual_distance.weight.reshape(1, self.head_size, 1)
        tree_attn_bias[:, :, 1:, 0] = tree_attn_bias[:, :, 1:, 0] + t
        tree_attn_bias[:, :, 0, :] = tree_attn_bias[:, :, 0, :] + t

        n_node = all_node_embedded.size()[2]
        index_attn_bias = index_attn_bias.reshape(-1, *index_attn_bias.shape[-2:]) # [n_batch * n_node, n_index, n_index]
        index_attn_bias = index_attn_bias.unsqueeze(1).repeat(1, self.head_size, 1, 1)    # [n_batch * n_node, n_head, n_index, n_index]

        # transfomrer encoder
        x = self.input_dropout(all_node_embedded) # [n_batch, n_index, n_node, hidden_dim]
        for enc_layer1, enc_layer2, comb_layer in zip(self.encoder_layers1, self.encoder_layers2, self.comb_layers):
            x1 = x.transpose(1, 2)   # [n_batch, n_node, n_index, hidden_dim]
            x1 = x1.reshape(n_batch * n_node, n_index, self.hidden_dim) # [n_batch*n_node, n_index, hidden_dim]

            x2 = x.reshape(n_batch * n_index, n_node, self.hidden_dim) # [n_batch*n_index, n_node, hidden_dim]

            x1 = enc_layer1(x1, index_attn_bias)
            x2 = enc_layer2(x2, tree_attn_bias)

            x1 = x1.reshape(n_batch, n_node, n_index, self.hidden_dim // 2) # [n_batch, n_node, n_index, hidden_dim/2]
            x1 = x1.transpose(1, 2) # [n_batch, n_index, n_node, hidden_dim]
            x2 = x2.reshape(n_batch, n_index, n_node, self.hidden_dim // 2) # [n_batch, n_index, n_node, hidden_dim/2]

            if self.disable_idx_attn:
                x1 = torch.zeros_like(x1)
                
            y = torch.cat([x1, x2], dim=-1)  # [n_batch, n_index, n_node, hidden_dim]
            y = self.attn_output_layer(y)

            x = x + y

            y = self.ffn_norm(x)
            y = self.ffn(y)
            y = self.ffn_dropout(y)
            x = x + y


        output = self.final_ln(x) # [n_batch, n_index, n_node, hidden_dim]
        return output


class Eddie(nn.Module):
    def __init__(self, emb_dim=32, ffn_dim=32, head_size=4, \
                 dropout=0.1, attention_dropout_rate=0.1, n_layers=4, \
                 pred_hid=128, hidden_dim=128, \
                 max_sort_col_num=5, max_output_col_num=5, max_predicate_num=120, \
                 predicate_types=['filter', 'join', 'index_cond'],
                 disable_idx_attn=False,
                 clip_label = True
                 ):

        super(Eddie, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.head_size = head_size
        
        self.operator_embedding = nn.Embedding(32, emb_dim)
        self.cost_embedding = nn.Embedding(30, emb_dim, padding_idx=0)
        self.rows_embedding = nn.Embedding(30, emb_dim, padding_idx=0)
        self.sort_consistent_embedding = nn.Embedding(3, emb_dim, padding_idx=0)
        self.col_pos_embedding = nn.Embedding(128, emb_dim, padding_idx=0)
        self.col_type_embedding = nn.Embedding(64, emb_dim, padding_idx=0)
        # self.col_stat_project = nn.Linear(3, emb_dim)
        self.col_stat_project = nn.Linear(2, emb_dim)
        
        self.compare_operator_embedding = nn.Embedding(20, emb_dim, padding_idx=0)
        self.selectivity_embedding = nn.Embedding(22, emb_dim, padding_idx=0)
        
        self.predicate_combined_embedded_dim =  self.col_pos_embedding.embedding_dim \
                                        + self.compare_operator_embedding.embedding_dim \
                                        + self.col_pos_embedding.embedding_dim \
                                        + self.selectivity_embedding.embedding_dim
                                        
        self.logical_operator_embedding = nn.Embedding(20, self.predicate_combined_embedded_dim, padding_idx=0)
        self.predicate_emb_project = nn.Linear(self.predicate_combined_embedded_dim, hidden_dim)
        
        self.max_predicate_num = max_predicate_num
        self.simple_predicate_project = nn.Linear(self.hidden_dim*max_predicate_num, hidden_dim)
                                        
        node_combined_embedded_dim = self.operator_embedding.embedding_dim \
                                + 2 * self.cost_embedding.embedding_dim \
                                + self.sort_consistent_embedding.embedding_dim \
                                + max_sort_col_num * self.col_pos_embedding.embedding_dim \
                                + max_output_col_num * self.col_pos_embedding.embedding_dim \
                                + self.hidden_dim
            
        self.node_emb_project = nn.Linear(node_combined_embedded_dim, hidden_dim)
        
        self.super_index_embedding = nn.Embedding(1, hidden_dim)
        
        self.node_height_embedding = nn.Embedding(64, hidden_dim, padding_idx=0)
        self.node_tree_bias_attn = NodeIndexAttention(disable_idx_attn=disable_idx_attn)

        self.predicate_height_embedding = nn.Embedding(64, hidden_dim, padding_idx=0)

        self.predicate_types = predicate_types
        self.predicate_attn_layers = nn.ModuleDict()
        for predicate_type in predicate_types:
            self.predicate_attn_layers[predicate_type] = TreeBiasAttention(n_layers=4)

        self.pred = Prediction(hidden_dim, pred_hid, clip_label=clip_label)
        
    def forward(self, features):
        n_batch, n_index, n_node = features['operator'].size()
        device = features['operator'].device
        
        predicate_feat_added = torch.zeros((n_batch, n_index, n_node, self.predicate_combined_embedded_dim), dtype=torch.float, device=device)
        for pt in self.predicate_types:
            predicate_tree_cnt = features[f'{pt}_logical_operator'].size()[0]
            predicate_feat = torch.zeros((n_batch*n_index*n_node, self.predicate_combined_embedded_dim), dtype=torch.float, device=device)
            
            if predicate_tree_cnt > 0:
                # predicate feature embedding
                predicate_logical_operator_embedded = self.logical_operator_embedding(features[f'{pt}_logical_operator'])
                predicate_compare_operator_embedded = self.compare_operator_embedding(features[f'{pt}_compare_operator'])
                predicate_selectivity_embedded = self.selectivity_embedding(features[f'{pt}_selectivity'])
                
                # predicate left col embedding
                predicate_left_col_pos_embedded = self.col_pos_embedding(features[f'{pt}_left_col_pos'])
                predicate_left_col_type_embedded = self.col_type_embedding(features[f'{pt}_left_col_type'])
                # predicate_left_col_stat_embedded = self.col_stat_project(torch.stack([features[f'{pt}_left_col_dist_cnt'], features[f'{pt}_left_col_dist_frac'], features[f'{pt}_left_col_null_frac']], dim=-1))
                predicate_left_col_stat_embedded = self.col_stat_project(torch.stack([features[f'{pt}_left_col_dist_frac'], features[f'{pt}_left_col_null_frac']], dim=-1))
                predicate_left_col_embedded = predicate_left_col_pos_embedded + predicate_left_col_type_embedded + predicate_left_col_stat_embedded
                
                # predicate right col embedding
                predicate_right_col_pos_embedded = self.col_pos_embedding(features[f'{pt}_right_col_pos'])
                predicate_right_col_type_embedded = self.col_type_embedding(features[f'{pt}_right_col_type'])
                # predicate_right_col_stat_embedded = self.col_stat_project(torch.stack([features[f'{pt}_right_col_dist_cnt'], features[f'{pt}_right_col_dist_frac'], features[f'{pt}_right_col_null_frac']], dim=-1))
                predicate_right_col_stat_embedded = self.col_stat_project(torch.stack([features[f'{pt}_right_col_dist_frac'], features[f'{pt}_right_col_null_frac']], dim=-1))
                predicate_right_col_embedded = predicate_right_col_pos_embedded + predicate_right_col_type_embedded + predicate_right_col_stat_embedded
                
                predicate_combined_embedded = torch.cat([
                    predicate_left_col_embedded,
                    predicate_compare_operator_embedded,
                    predicate_selectivity_embedded,
                    predicate_right_col_embedded,
                ], dim=-1)
                predicate_combined_embedded += predicate_logical_operator_embedded
                predicate_embedded = F.leaky_relu(self.predicate_emb_project(predicate_combined_embedded))
                
                # predicate wise attention (tree-bias attention)
                predicate_height_embedded = self.predicate_height_embedding(features[f'{pt}_heights'])
                predicate_embedded = predicate_embedded + predicate_height_embedded

                predicate_attn_output = self.predicate_attn_layers[pt](predicate_embedded, features[f'{pt}_rel_pos_mat'], features[f'{pt}_tree_attn_bias'])
                super_predicate_output = predicate_attn_output[:, 0, :]
                    
                predicate_mask = features[f'{pt}_predicate_tree_mask'].reshape(n_batch*n_index*n_node)
                predicate_feat[predicate_mask == 1] = super_predicate_output

            predicate_feat = predicate_feat.reshape(n_batch, n_index, n_node, -1)
            predicate_feat_added += predicate_feat

        # sort col embedding
        sort_col_pos_embedded = self.col_pos_embedding(features['sort_col_pos']) # [n_batch, n_index, n_node, n_sort_col, emb_dim]
        sort_col_type_embedded = self.col_type_embedding(features['sort_col_type']) # [n_batch, n_index, n_node, n_sort_col, emb_dim]
        # sort_col_stat_embedded = self.col_stat_project(torch.stack([features['sort_col_dist_cnt'], features['sort_col_dist_frac'], features['sort_col_null_frac']], dim=-1))
        sort_col_stat_embedded = self.col_stat_project(torch.stack([features['sort_col_dist_frac'], features['sort_col_null_frac']], dim=-1))
        sort_col_embedded = sort_col_pos_embedded + sort_col_type_embedded + sort_col_stat_embedded

        # output col embedding
        output_col_pos_embedded = self.col_pos_embedding(features['output_col_pos']) # [n_batch, n_index, n_node, n_output_col, emb_dim]
        output_col_type_embedded = self.col_type_embedding(features['output_col_type']) # [n_batch, n_index, n_node, n_output_col, emb_dim]
        # output_col_stat_embedded = self.col_stat_project(torch.stack([features['output_col_dist_cnt'], features['output_col_dist_frac'], features['output_col_null_frac']], dim=-1))
        output_col_stat_embedded = self.col_stat_project(torch.stack([features['output_col_dist_frac'], features['output_col_null_frac']], dim=-1))
        output_col_embedded = output_col_pos_embedded + output_col_type_embedded + output_col_stat_embedded
        
        # node feature embedding
        operator_embedded = self.operator_embedding(features['operator'])
        sort_consistent_embedded = self.sort_consistent_embedding(features['sort_consistent'])
        cost_embedded = self.cost_embedding(features['cost'])
        rows_embedded = self.rows_embedding(features['rows'])

        # Concatenate all embeddings along the last dimension
        combined_embedded = torch.cat([
            operator_embedded,
            cost_embedded,
            rows_embedded,
            sort_consistent_embedded,
            sort_col_embedded.reshape(*sort_col_embedded.shape[:-2], -1),
            output_col_embedded.reshape(*output_col_embedded.shape[:-2], -1),
            predicate_feat_added
        ], dim=-1)
        
        node_embedded = F.leaky_relu(self.node_emb_project(combined_embedded)) # [n_batch, n_index, n_node, hidden_dim]
        node_embedded *= features['node_padding_mask'].unsqueeze(-1) # node_padding_mask: [n_batch, n_index, n_node]
    
        # node wise attention (tree-bias attention)
        node_height_embedded = self.node_height_embedding(features['node_heights'])
        node_heights_expanded = node_height_embedded.unsqueeze(1).repeat(1, n_index, 1, 1) # [n_batch, n_index, n_node, hidden_dim]
        node_rel_pos_mat_expanded = features['node_rel_pos_mat'].unsqueeze(1).repeat(1, n_index, 1, 1) # [n_batch, n_index, n_node+1, n_node+1]
        node_tree_attn_bias_expanded = features['node_tree_attn_bias'].unsqueeze(1).repeat(1, n_index, 1, 1) # [n_batch, n_index, n_node+1, n_node+1]
        node_attn_output = self.node_tree_bias_attn(node_embedded, node_heights_expanded, node_rel_pos_mat_expanded, node_tree_attn_bias_expanded, features['index_attn_bias'])
        
        super_node_output = node_attn_output[:, :, 0, :] # [n_batch, n_index, hidden_dim]
        
        index_padding_mask, _ = torch.max(features['node_padding_mask'] , dim=2) # [n_batch, n_index]
        super_node_output *= index_padding_mask.unsqueeze(-1)
        index_nums = index_padding_mask.sum(dim=1) # [n_batch]
        sum_pooled = super_node_output.sum(dim=1) # [n_batch, hidden_dim]
        avg_pooled = sum_pooled / index_nums.unsqueeze(1)
        
        result = self.pred(avg_pooled)
        
        return result
    