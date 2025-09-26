import torch
from torch.utils.data import Dataset

from aimeetsai.aimai_feat import *
import numpy as np


class AiMAiDataset(Dataset):
    def __init__(self, sample_data):
        
        processed_samples = []
        
        for sample in sample_data:
            indexes = sample["indexes"]
            label = sample["label"]
            init_plan = sample["plan_tree"]
            hypo_plan = sample["plan_hypo"]

            pair_diff_normalized_matrix = process_plan_pair(init_plan, hypo_plan)
                                        
            processed_samples.append((pair_diff_normalized_matrix, label))
        
        self.data = processed_samples

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def collate_fn4aimai(batch):
    data = [sample[0] for sample in batch]
    label = [sample[1] for sample in batch]
    data = torch.tensor(np.array(data), dtype=torch.float)
    label = torch.tensor(label, dtype=torch.float)
    
    return ({"feats": data}, label)
