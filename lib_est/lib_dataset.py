import torch
from torch.utils.data import Dataset

from lib_est.get_plan_info import tra_plan_ite
import numpy as np


ops_join_dict = {"Hash Join": 0, "Hash Right Join": 0, "Hash Left Join": 0, "Merge Join": 1, "Nested Loop": 2}
ops_sort_dict = {"Sort": 0}
ops_group_dict = {"Aggregate": 0, "Group": 1}
ops_scan_dict = {"Bitmap Heap Scan": 0, "Bitmap Index Scan": 1, "Index Scan": 2, "Index Only Scan": 3, "Seq Scan": 4}


class LibDataset(Dataset):
    def __init__(self, sample_data, stats):
        # MODIFIED: This function is adapted from the implementation in [Index_EAB].
        # Source code: https://github.com/XMUDM/Index_EAB/blob/main/index_advisor_selector/index_benefit_estimation/index_cost_lib/pre_lib_data.py
        
        processed_samples = []
        
        for sample in sample_data:
            indexes = sample["indexes"]
            label = sample["label"]
            nodes = tra_plan_ite(sample["plan_tree"])

            index_ops = list()
            for ind in indexes:
                tbl = ind[0].split(".")[0]
                cols = [tbl_col.split(".")[1] for tbl_col in ind]
                # tbl = ind.split("#")[0]
                # cols = ind.split("#")[1].split(",")

                for no, col in enumerate(cols):
                    for node in nodes:
                        if col in str(node["detail"]):
                            # 1. operation information (5)
                            # join, sort, group, scan_range, scan_equal
                            vec = [0 for _ in range(5)]
                            typ = node["type"]
                            if typ in ops_join_dict:
                                vec[0] = 1
                            elif typ in ops_sort_dict:
                                vec[1] = 1
                            elif typ in ops_group_dict:
                                vec[2] = 1
                            elif typ in ops_scan_dict:
                                # (1005): to be improved. columns with the same name.
                                if f"{col} =" in str(node["detail"]):
                                    vec[3] = 1
                                else:
                                    vec[4] = 1

                            # 2. database statistics (4)
                            # card = 0 if node["detail"]["Plan Rows"] == 0 else np.log(node["detail"]["Plan Rows"])
                            if "Actual Rows" not in node["detail"] or node["detail"]["Actual Rows"] == 0:
                                card = 0
                            else:
                                card = np.log(node["detail"]["Actual Rows"])
                            row = np.log(stats[f"{tbl}.{col}"]["rows"])
                            null = stats[f"{tbl}.{col}"]["null_frac"]
                            dist = stats[f"{tbl}.{col}"]["n_distinct"] / stats[f"{tbl}.{col}"]["rows"]

                            vec.extend([card, row, null, dist])

                            # 3. index information (3)
                            if len(col) == 1:
                                vec.extend([1, 0, 0])
                            else:
                                vec.extend([0, 1, no + 1])

                            index_ops.append(vec)
                            
            processed_samples.append((index_ops, label))
        
        self.data = processed_samples

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def collate_fn4lib(batch):
    # MODIFIED: This function is adapted from the original implementation in [Learned-Index-Benefits].
    # Source code: https://github.com/JC-Shi/Learned-Index-Benefits/blob/main/LIB_by_query.ipynb
    
    data = [sample[0] for sample in batch]
    label = [sample[1] for sample in batch]
    
    # Find the maximum number of index optimizable operations
    max_l = 0
    min_l = 999
    for i in range(0,len(data)):
        max_l = max(max_l, len(data[i]))
        min_l = min(min_l, len(data[i]))

    # Pad data to faciliate batch training/testing
    pad_data = []
    mask = []
    pad_element = [0 for i in range(0,12)]
    for data_point in data:

        new_data = []
        point_mask = [0 for i in range(0,max_l)]

        for j in range(0,len(data_point)):
            new_data.append(data_point[j])
            point_mask[j] = 1

        if max_l - len(data_point) > 0:
            for k in range(0,max_l-len(data_point)):
                new_data.append(pad_element)

        pad_data.append(new_data)
        mask.append(point_mask)
    
    pad_data = torch.tensor(pad_data, dtype=torch.float)
    mask = torch.tensor(mask, dtype=torch.int)
    label = torch.tensor(label, dtype=torch.float)
    
    return ({"feats": pad_data, "mask": mask}, label)
