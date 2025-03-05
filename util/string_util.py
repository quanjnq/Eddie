import hashlib
import pickle


def string_to_md5(string):
    md5_val = hashlib.md5(string.encode('utf8')).hexdigest()
    return md5_val

def get_pretrain_dataset_name(pretrain_data_list_dict):
    dataset_name = "pretrain"
    for dataset_type in pretrain_data_list_dict:
        data_list = pretrain_data_list_dict[dataset_type]
        if len(data_list) == 0:
            continue
        dataset_type_name = f"{dataset_type}"
        for fp in data_list:
            with open(fp, "rb") as f:
                fplen = len(pickle.load(f))
            dataset_type_name += "_"+str(fplen)
        dataset_name += "_"+dataset_type_name

    
    slen = 0
    for dataset_type in pretrain_data_list_dict:
        data_list = pretrain_data_list_dict[dataset_type]
        if len(data_list) == 0:
            continue
        for fp in data_list:
            with open(fp, "rb") as f:
                slen += min(len(pickle.load(f)), 10000)
    dataset_name = f"extdata_cl{len(dataset_name)}" if len(dataset_name) > 100 else dataset_name
    return dataset_name+f"_len{slen}"+f"_{len(pretrain_data_list_dict)}db"