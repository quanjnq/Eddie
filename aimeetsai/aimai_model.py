import torch
from torch.utils.data import Dataset
import json
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F


class AiMAiPartiallyConnectedLayers(nn.Module):
    def __init__(self, num_channels = 5, num_node_types = 26, hid_units = 64):
        super(AiMAiPartiallyConnectedLayers, self).__init__()
        
        self.num_channels = num_channels
        self.num_node_types = num_node_types
        self.hid_units = hid_units
        self.per_key_networks = nn.ModuleDict()
        
        for node_type_idx in range(num_node_types):
            network = nn.Sequential(
                nn.Linear(num_channels, hid_units),
                nn.Tanh(),
                nn.Linear(hid_units, hid_units),
                nn.Tanh(),
                nn.Linear(hid_units, 1)
            )
            self.per_key_networks[str(node_type_idx)] = network
        

    def forward(self, features):
        
        batch_size = features.shape[0]
        
        
        node_outputs = []
        
        
        for node_type_idx in range(self.num_node_types):
            
            node_features = features[:, node_type_idx, :]  
            
            
            node_output = self.per_key_networks[str(node_type_idx)](node_features)
            node_outputs.append(node_output)
        
        
        combined_features = torch.cat(node_outputs, dim=1)
        
        return combined_features
    

class AiMAiFullyConnectedLayers(nn.Module):
    def __init__(self, in_feature = 26, hid_units = 64):
        super(AiMAiFullyConnectedLayers, self).__init__()
        
        
        layers = []
        current_dim = in_feature
        
        for i in range(12):
            layers.append(nn.Linear(current_dim, hid_units))
            layers.append(nn.Tanh())
            current_dim = hid_units
        
        self.network = nn.Sequential(*layers)

    def forward(self, features):
        out = self.network(features)
        return out
    

class AiMAiPrediction(nn.Module):
    def __init__(self, in_feature = 64, out_feature = 128):
        super(AiMAiPrediction, self).__init__()
        
        self.out_mlp1 = nn.Linear(in_feature, out_feature)
        self.mid_mlp1 = nn.Linear(out_feature, out_feature)
        self.out_mlp2 = nn.Linear(out_feature, 1)

    def forward(self, features):
        hid = F.relu(self.out_mlp1(features))
        mid = F.relu(self.mid_mlp1(hid))
        out = torch.sigmoid(self.out_mlp2(mid))
        return out


class AiMAiModel(nn.Module):
    def __init__(self, num_channels = 5, num_node_types = 26, hid_units = 64):
        
        super(AiMAiModel,self).__init__()
        
        
        self.partially_connected = AiMAiPartiallyConnectedLayers(
            num_channels=num_channels, 
            num_node_types=num_node_types, 
            hid_units=hid_units
        )
        self.fully_connected = AiMAiFullyConnectedLayers(in_feature = num_node_types, hid_units = hid_units)
        
        self.pred = AiMAiPrediction(in_feature = hid_units) 
        
    def forward(self, batched_data):
        x = batched_data['feats'] # [batch_size, channel_size, node_type_size]
        x = x.transpose(1, 2)  # [batch_size, node_type_size, channel_size]
        x = self.partially_connected(x)  # [batch_size, node_type_size]
        x = self.fully_connected(x)  # [batch_size, hid_units]
        return self.pred(x)
