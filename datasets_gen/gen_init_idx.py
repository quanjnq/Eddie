import pickle
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import logging


def is_real_sub(inds, best_inds):
    if len(inds) >= len(best_inds):
        return False
    for ind in inds:
        if ind not in best_inds:
            return False
    return True


def super_set(indcfg, indcfgs):
    superset = set()
    for iptcfg in indcfgs:
        if is_real_sub(indcfg, iptcfg):
            superset.add(iptcfg)
    return superset
    
    
def compose_init_indexes_sample(sub_sample, sample):
    _, init_indexes = sub_sample[1]
    orig_times, orig_plan = sub_sample[2]
    orig_times_init_ind, orig_plan_init_ind = sub_sample[3]
    if orig_times_init_ind is None and orig_plan_init_ind is None:
        orig_times_init_ind = orig_times
        orig_plan_init_ind = orig_plan
    _, indexes = sample[1]
    query = sample[0]
    with_index_runtimes, with_index_plan = sample[3]
    sample_with_init_indexes = (query, (init_indexes, indexes), (orig_times_init_ind, orig_plan_init_ind), (with_index_runtimes, with_index_plan))
    return sample_with_init_indexes


def gen_with_init_idx_samples(base_sample_path, save_path):
    with open(base_sample_path, "rb") as f:
        smpls = pickle.load(f)
    q2smpls = {}
    for sp in smpls:
        q = sp[0]
        if q not in q2smpls:
            q2smpls[q] = []
        q2smpls[q].append(sp)
        
    with_init_ind_samples = []

    for q in q2smpls:
        indcfgs = set([sp[1][1] for sp in q2smpls[q]])
        indcfg2sample = dict()
        for sp in q2smpls[q]:
            indcfg2sample[sp[1][1]] = sp
        labels_candidate = []
        candidate = list(indcfgs)[-1]
        for i, indcfg in enumerate(indcfgs):
            labels = []
            input_indcfgs = super_set(indcfg, indcfgs)
            for ipt_indcfg in input_indcfgs:

                new_sample = compose_init_indexes_sample(indcfg2sample[indcfg], indcfg2sample[ipt_indcfg])
                if new_sample is None or (new_sample[2][0] is None and new_sample[2][1] is None):
                    continue
                with_init_ind_samples.append(new_sample)
                ind_time = new_sample[3][0] if new_sample[3][0] else new_sample[2][0]
                labels.append((new_sample[2][0] - ind_time) / new_sample[2][0])
                if candidate == ipt_indcfg:
                    
                    labels_candidate.append((new_sample[2][0] - ind_time) / new_sample[2][0])
    
    with_init_ind_samples = with_init_ind_samples +  smpls
    
    dst_dir = os.path.dirname(save_path)
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
    with open(save_path, "wb") as f:
        pickle.dump(with_init_ind_samples, f)
        logging.info(f"wo dataset size: {len(smpls)}")
        logging.info(f"w dataset size: {len(with_init_ind_samples)}")
        logging.info(f"Saved dataset: {save_path}")
        
    return with_init_ind_samples
