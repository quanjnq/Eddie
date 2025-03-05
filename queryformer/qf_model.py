"""
This code is based on the implementation from the original repository of [QueryFormer].
Source code: https://github.com/zhaoyue-ntu/QueryFormer/blob/main/model/model.py

For detailed changes, look for comments marked as 'MODIFIED' in the code.
"""

import torch
from torch.utils.data import Dataset
import json
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F

class Prediction(nn.Module):
    def __init__(self, in_feature = 69, hid_units = 256, contract = 1, mid_layers = True, res_con = True):
        super(Prediction, self).__init__()
        self.mid_layers = mid_layers
        self.res_con = res_con
        
        self.out_mlp1 = nn.Linear(in_feature, hid_units)

        self.mid_mlp1 = nn.Linear(hid_units, hid_units//contract)
        self.mid_mlp2 = nn.Linear(hid_units//contract, hid_units)

        self.out_mlp2 = nn.Linear(hid_units, 1)

    def forward(self, features):
        
        hid = F.relu(self.out_mlp1(features))
        if self.mid_layers:
            mid = F.relu(self.mid_mlp1(hid))
            mid = F.relu(self.mid_mlp2(mid))
            if self.res_con:
                hid = hid + mid
            else:
                hid = mid
        out = torch.sigmoid(self.out_mlp2(hid))

        return out

        
class FeatureEmbed(nn.Module):
    def __init__(self, embed_size=32, tables=26, types=30, joins=1920, columns=430, \
                 ops=7, use_sample=True, use_hist=True, bin_number=50): # MODIFIED: default value
        super(FeatureEmbed, self).__init__()
        
        self.use_sample = use_sample
        self.embed_size = embed_size        
        
        self.use_hist = use_hist
        self.bin_number = bin_number
        
        self.typeEmbed = nn.Embedding(types, embed_size)
        self.tableEmbed = nn.Embedding(tables, embed_size)
        
        self.columnEmbed = nn.Embedding(columns, embed_size)
        self.opEmbed = nn.Embedding(ops, embed_size//8)

        self.linearFilter2 = nn.Linear(embed_size+embed_size//8+1, embed_size+embed_size//8+1)
        self.linearFilter = nn.Linear(embed_size+embed_size//8+1, embed_size+embed_size//8+1)

        self.linearType = nn.Linear(embed_size, embed_size)
        
        self.linearJoin = nn.Linear(embed_size, embed_size)
        
        self.linearSample = nn.Linear(1000, embed_size)
        
        self.linearHist = nn.Linear(bin_number, embed_size)

        self.joinEmbed = nn.Embedding(joins, embed_size)
        
        if use_hist:
            self.project = nn.Linear(embed_size*5 + embed_size//8+1, embed_size*5 + embed_size//8+1)
        else:
            self.project = nn.Linear(embed_size*4 + embed_size//8+1, embed_size*4 + embed_size//8+1)
    
    # input: B by 14 (type, join, f1, f2, f3, mask1, mask2, mask3)
    def forward(self, feature):

        typeId, joinId, filtersId, filtersMask, hists, table_sample = torch.split(feature,(1,1,9,3,self.bin_number*3,1001), dim = -1)
        
        typeEmb = self.getType(typeId)
        joinEmb = self.getJoin(joinId)
        filterEmbed = self.getFilter(filtersId, filtersMask)
        
        histEmb = self.getHist(hists, filtersMask)
        tableEmb = self.getTable(table_sample)
    
        if self.use_hist:
            final = torch.cat((typeEmb, filterEmbed, joinEmb, tableEmb, histEmb), dim = 1)
        else:
            final = torch.cat((typeEmb, filterEmbed, joinEmb, tableEmb), dim = 1)
        final = F.leaky_relu(self.project(final))
        
        return final
    
    def getType(self, typeId):
        emb = self.typeEmbed(typeId.long())

        return emb.squeeze(1)
    
    def getTable(self, table_sample):
        table, sample = torch.split(table_sample,(1,1000), dim = -1)
        emb = self.tableEmbed(table.long()).squeeze(1)
        
        if self.use_sample:
            emb += self.linearSample(sample)
        return emb
    
    def getJoin(self, joinId):
        emb = self.joinEmbed(joinId.long())

        return emb.squeeze(1)

    def getHist(self, hists, filtersMask):
        # batch * 50 * 3
        histExpand = hists.view(-1,self.bin_number,3).transpose(1,2)
        
        emb = self.linearHist(histExpand)
        emb[~filtersMask.bool()] = 0.  # mask out space holder
        
        ## avg by # of filters
        num_filters = torch.sum(filtersMask,dim = 1)
        total = torch.sum(emb, dim = 1)
        avg = total / num_filters.view(-1,1)
        
        return avg
        
    def getFilter(self, filtersId, filtersMask):
        ## get Filters, then apply mask
        filterExpand = filtersId.view(-1,3,3).transpose(1,2)
        colsId = filterExpand[:,:,0].long()
        opsId = filterExpand[:,:,1].long()
        vals = filterExpand[:,:,2].unsqueeze(-1) # b by 3 by 1
        
        # b by 3 by embed_dim
        
        col = self.columnEmbed(colsId)
        op = self.opEmbed(opsId)
        
        concat = torch.cat((col, op, vals), dim = -1)
        concat = F.leaky_relu(self.linearFilter(concat))
        concat = F.leaky_relu(self.linearFilter2(concat))
        
        ## apply mask
        concat[~filtersMask.bool()] = 0.
        
        ## avg by # of filters
        num_filters = torch.sum(filtersMask,dim = 1)
        total = torch.sum(concat, dim = 1)
        avg = total / num_filters.view(-1,1)
                
        return avg
    
#     def get_output_size(self):
#         size = self.embed_size * 5 + self.embed_size // 8 + 1
#         return size



class QueryFormer(nn.Module):
    def __init__(self, emb_size = 32 ,ffn_dim = 32, head_size = 8, \
                 dropout = 0.1, attention_dropout_rate = 0.1, n_layers = 8, \
                 use_sample = True, use_hist = True, bin_number = 50, \
                 pred_hid = 256
                ):
        
        super(QueryFormer,self).__init__()
        if use_hist:
            hidden_dim = emb_size * 5 + emb_size //8 + 1
        else:
            hidden_dim = emb_size * 4 + emb_size //8 + 1
        self.hidden_dim = hidden_dim
        self.head_size = head_size
        self.use_sample = use_sample
        self.use_hist = use_hist

        self.rel_pos_encoder = nn.Embedding(192, head_size, padding_idx=0) # MODIFIED: num_embeddings 64->192

        self.height_encoder = nn.Embedding(64, hidden_dim, padding_idx=0)
        
        self.input_dropout = nn.Dropout(dropout)
        encoders = [EncoderLayer(hidden_dim, ffn_dim, dropout, attention_dropout_rate, head_size)
                    for _ in range(n_layers)]
        self.layers = nn.ModuleList(encoders)
        
        self.final_ln = nn.LayerNorm(hidden_dim)
        
        self.super_token = nn.Embedding(1, hidden_dim)
        self.super_token_virtual_distance = nn.Embedding(1, head_size)
        
        self.embbed_layer = FeatureEmbed(emb_size, use_sample = use_sample, use_hist = use_hist, bin_number = bin_number)
        
        # MODIFIED: Output only the plan representation in this module; the prediction layer is not needed.
        # self.pred = Prediction(hidden_dim, pred_hid)

        # # if multi-task
        # self.pred2 = Prediction(hidden_dim, pred_hid)
        
    def forward(self, attn_bias, rel_pos, x, heights): # MODIFIED: Changed input format to integrate with a unified process. 
        n_batch, n_node = x.size()[:2]
        tree_attn_bias = attn_bias.clone()
        tree_attn_bias = tree_attn_bias.unsqueeze(1).repeat(1, self.head_size, 1, 1) 
        
        # rel pos
        rel_pos_bias = self.rel_pos_encoder(rel_pos).permute(0, 3, 1, 2) # [n_batch, n_node, n_node, n_head] -> [n_batch, n_head, n_node, n_node]
        tree_attn_bias[:, :, 1:, 1:] = tree_attn_bias[:, :, 1:, 1:] + rel_pos_bias


        # reset rel pos here
        t = self.super_token_virtual_distance.weight.view(1, self.head_size, 1)
        tree_attn_bias[:, :, 1:, 0] = tree_attn_bias[:, :, 1:, 0] + t
        tree_attn_bias[:, :, 0, :] = tree_attn_bias[:, :, 0, :] + t
        
        x_view = x.view(-1, 1165)
        node_feature = self.embbed_layer(x_view).view(n_batch,-1, self.hidden_dim)
        
        # -1 is number of dummy
        
        node_feature = node_feature + self.height_encoder(heights)
        super_token_feature = self.super_token.weight.unsqueeze(0).repeat(n_batch, 1, 1)
        super_node_feature = torch.cat([super_token_feature, node_feature], dim=1)        
        
        # transfomrer encoder
        output = self.input_dropout(super_node_feature)
        for enc_layer in self.layers:
            output = enc_layer(output, tree_attn_bias)
        output = self.final_ln(output)
        
        return output[:,0,:]


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
    def __init__(self, hidden_size, attention_dropout_rate, head_size):
        super(MultiHeadAttention, self).__init__()

        self.head_size = head_size

        self.att_size = att_size = hidden_size // head_size
        self.scale = att_size ** -0.5

        self.linear_q = nn.Linear(hidden_size, head_size * att_size)
        self.linear_k = nn.Linear(hidden_size, head_size * att_size)
        self.linear_v = nn.Linear(hidden_size, head_size * att_size)
        self.att_dropout = nn.Dropout(attention_dropout_rate)

        self.output_layer = nn.Linear(head_size * att_size, hidden_size)

    def forward(self, q, k, v, attn_bias=None):
        orig_q_size = q.size()

        d_k = self.att_size
        d_v = self.att_size
        batch_size = q.size(0)

        # head_i = Attention(Q(W^Q)_i, K(W^K)_i, V(W^V)_i)
        q = self.linear_q(q).view(batch_size, -1, self.head_size, d_k)
        k = self.linear_k(k).view(batch_size, -1, self.head_size, d_k)
        v = self.linear_v(v).view(batch_size, -1, self.head_size, d_v)

        q = q.transpose(1, 2)                  # [b, h, q_len, d_k]
        v = v.transpose(1, 2)                  # [b, h, v_len, d_v]
        k = k.transpose(1, 2).transpose(2, 3)  # [b, h, d_k, k_len]

        # Scaled Dot-Product Attention.
        # Attention(Q, K, V) = softmax((QK^T)/sqrt(d_k))V
        q = q * self.scale
        x = torch.matmul(q, k)  # [b, h, q_len, k_len]
        if attn_bias is not None:
            x = x + attn_bias

        x = torch.softmax(x, dim=3)
        x = self.att_dropout(x)
        x = x.matmul(v)  # [b, h, q_len, attn]

        x = x.transpose(1, 2).contiguous()  # [b, q_len, h, attn]
        x = x.view(batch_size, -1, self.head_size * d_v)

        x = self.output_layer(x)

        assert x.size() == orig_q_size
        return x


class EncoderLayer(nn.Module):
    def __init__(self, hidden_size, ffn_size, dropout_rate, attention_dropout_rate, head_size):
        super(EncoderLayer, self).__init__()

        self.self_attention_norm = nn.LayerNorm(hidden_size)
        self.self_attention = MultiHeadAttention(hidden_size, attention_dropout_rate, head_size)
        self.self_attention_dropout = nn.Dropout(dropout_rate)

        self.ffn_norm = nn.LayerNorm(hidden_size)
        self.ffn = FeedForwardNetwork(hidden_size, ffn_size, dropout_rate)
        self.ffn_dropout = nn.Dropout(dropout_rate)

    def forward(self, x, attn_bias=None):
        y = self.self_attention_norm(x)
        y = self.self_attention(y, y, y, attn_bias)
        y = self.self_attention_dropout(y)
        x = x + y

        y = self.ffn_norm(x)
        y = self.ffn(y)
        y = self.ffn_dropout(y)
        x = x + y
        return x


# MODIFIED: add IndexesEmbed
class IndexesEmbed(nn.Module):
    def __init__(self, embed_size=32, columns=600, position_num=15,hid_dim=256):
        super(IndexesEmbed, self).__init__()

        self.embed_size=embed_size
        self.position_num=position_num
        self.indexEmb = nn.Embedding(columns, embed_size,padding_idx=0)

        self.project = nn.Linear(embed_size*position_num , embed_size*position_num )
        self.mid1=nn.Linear(embed_size*position_num,hid_dim)
        self.output=nn.Linear(hid_dim,embed_size)

    def forward(self, feature):
        final = self.getIndex(feature)
        final = final.reshape(final.shape[0],-1)
        final = F.leaky_relu(self.project(final))
        final = F.leaky_relu(self.mid1(final))
        final = F.leaky_relu(self.output(final))
        
        return final

    def getIndex(self, indexID):
        emb = self.indexEmb(indexID.long())
        return emb
    
    
    def get_output_size(self):
        size = self.embed_size
        return size


# MODIFIED: integrate QueryFormer with index feature
class QueryFormerWithIndex(nn.Module):
    def __init__(self, emb_size = 32 ,ffn_dim = 32, head_size = 8, \
                 dropout = 0.1, attention_dropout_rate = 0.1, n_layers = 8, \
                 use_sample = True, use_hist = True, bin_number = 50, \
                 pred_hid = 256
                ):
        
        super(QueryFormerWithIndex,self).__init__()
        
        self.queryformer = QueryFormer(emb_size,ffn_dim, head_size, \
                                        dropout, attention_dropout_rate, n_layers, \
                                        use_sample, use_hist, bin_number, \
                                        pred_hid)
        
        self.indexEmb = IndexesEmbed(emb_size)
        self.pred = Prediction(self.queryformer.hidden_dim + self.indexEmb.get_output_size(), pred_hid) 
        
    def forward(self, batched_data):
        x = batched_data['plan_x']
        attn_bias = batched_data['plan_attn_bias']
        rel_pos = batched_data['plan_rel_pos']
        heights = batched_data['plan_heights']
        
        output = self.queryformer(attn_bias, rel_pos, x, heights)
        index_output=self.indexEmb(batched_data['index_feat'])
        final_output = torch.cat([output, index_output], dim=1)
        return self.pred(final_output)


# MODIFIED: add AIMAIPrediction
class AIMAIPrediction(nn.Module):
    def __init__(self, in_feature = 69, hid_units = 256, contract = 1):
        super(AIMAIPrediction, self).__init__()
        
        self.out_mlp1 = nn.Linear(in_feature, hid_units)
        self.mid_mlp1 = nn.Linear(hid_units, hid_units//contract)
        self.mid_mlp2 = nn.Linear(hid_units//contract, hid_units)
        self.out_mlp2 = nn.Linear(hid_units, 1)

    def forward(self, features):
        hid = F.relu(self.out_mlp1(features))
        mid = F.relu(self.mid_mlp1(hid)+ hid)
        mid = F.relu(self.mid_mlp2(mid)+ mid)
        out = torch.sigmoid(self.out_mlp2(mid))
        return out
    
    
# MODIFIED: integrate QueryFormer with AIMeetsAI
class QueryFormerWithAIMAI(nn.Module):
    def __init__(self, emb_size = 32 ,ffn_dim = 32, head_size = 8, \
                 dropout = 0.1, attention_dropout_rate = 0.1, n_layers = 8, \
                 use_sample = True, use_hist = True, bin_number = 50, \
                 pred_hid = 256
                ):
        
        super(QueryFormerWithAIMAI,self).__init__()
        
        self.queryformer = QueryFormer(emb_size,ffn_dim, head_size, \
                                        dropout, attention_dropout_rate, n_layers, \
                                        use_sample, use_hist, bin_number, \
                                        pred_hid)
        
        self.pred = AIMAIPrediction(self.queryformer.hidden_dim, pred_hid) 
        
    def forward(self, batched_data):
        x = batched_data['plan_x']
        attn_bias = batched_data['plan_attn_bias']
        rel_pos = batched_data['plan_rel_pos']
        heights = batched_data['plan_heights']
        plan_emb = self.queryformer(attn_bias, rel_pos, x, heights)
        
        hypo_x = batched_data['plan_hypo_x']
        hypo_attn_bias = batched_data['plan_hypo_attn_bias']
        hypo_rel_pos = batched_data['plan_hypo_rel_pos']
        hypo_heights = batched_data['plan_hypo_heights']
        plan_hypo_emb = self.queryformer(hypo_attn_bias, hypo_rel_pos, hypo_x, hypo_heights)
        
        return self.pred(plan_emb - plan_hypo_emb)
