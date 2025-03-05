"""
This code is based on the implementation from the original repository of [Learned-Index-Benefits].
Source code: https://github.com/JC-Shi/Learned-Index-Benefits/blob/main/LIB_by_query.ipynb

For detailed changes, look for comments marked as 'MODIFIED' in the code.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import copy, math


# LIB consists of three parts: (1) feature extractor, (2) Encoder and (3) Prediction model.
# This notebook shows the design of the encoder and the prediction model.

def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
    
class SublayerConnection(nn.Module):
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))
    
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.4):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))
    
def attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    
    return torch.matmul(p_attn, value), p_attn

class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.4):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(1)
        nbatches = query.size(0)
        
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        x, self.attn = attention(query, key, value, mask=mask, 
                                 dropout=self.dropout)
        
        x = x.transpose(1, 2).contiguous() \
             .view(nbatches, -1, self.h * self.d_k)
        
        return self.linears[-1](x)
    
class Encoder(nn.Module):
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
        
    def forward(self, y, x, mask):
        for layer in self.layers:
            x = layer(y, x, mask)
        return self.norm(x)
    
class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, y, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(y, x, x, mask))
        
        return self.sublayer[1](x, self.feed_forward)
    
# Pooling by attention
class PoolingLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(PoolingLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = SublayerConnection(size, dropout)
        self.size = size

    def forward(self, y, x, mask):
        x = self.self_attn(y, x, x, mask)
        return self.sublayer(x, self.feed_forward)
    
def make_model(d_model, N, d_ff, h, dropout=0.4):
    
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model, dropout)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    
    model = Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N)
    pooling_model = Encoder(PoolingLayer(d_model, c(attn), c(ff), dropout), 1)
    
    return model, pooling_model

# Completed LIB
class self_attn_model(nn.Module):
    
    def __init__(self, encoder, pooling_model, ini_feats, encode_feats, hidden_dim):
        super(self_attn_model, self).__init__()
        
        self.encoder = encoder
        self.pma = pooling_model
        self.S = nn.Parameter(torch.Tensor(1,1,encode_feats))
        nn.init.xavier_uniform_(self.S)
        self.output1 = nn.Linear(encode_feats, hidden_dim, bias = True)
        self.output2 = nn.Linear(hidden_dim, 1, bias = True)
        self.linear1 = nn.Linear(ini_feats, encode_feats, bias = True)
        
    def forward(self, batch):
        # MODIFIED: Changed input format to integrate with a unified process. 
        batch_samples = batch['feats']
        batch_mask = batch['mask']
        
        batch_samples = F.relu(self.linear1(batch_samples))
        attn_output = self.encoder(batch_samples, batch_samples, batch_mask)
        attn_output = self.pma(self.S.repeat(attn_output.size(0),1,1), attn_output, batch_mask)
        hidden_rep = F.relu(self.output1(attn_output))
        out = torch.sigmoid(self.output2(hidden_rep))
        
        return out
