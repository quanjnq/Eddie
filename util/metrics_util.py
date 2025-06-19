import numpy as np
import math
from sklearn.metrics import mean_absolute_error
import logging
from collections import defaultdict


def calc_qerror(preds, labels):
    if len(preds) == 0:
        return {
        'q_50': 0,
        'q_90': 0,
        'q_95': 0,
        'q_mean': 0,
        'mae': 0
    }
    qerror = []
    for i in range(len(preds)):
        pred = max(preds[i], 0)
        label = max(labels[i], 0)
        qerror.append(max((pred + 1e-4) / (label + 1e-4), (label + 1e-4) / (pred + 1e-4)))
        
    e_50, e_90, e_95 = np.median(qerror), np.percentile(qerror, 90), np.percentile(qerror, 95)
    e_mean = np.mean(qerror)
    e_max = np.max(qerror)

    res = {
        'q_50': e_50,
        'q_90': e_90,
        'q_95': e_95,
        'q_mean': e_mean,
        # 'q_max': e_max,
        'mae': mean_absolute_error(preds, labels)
    }

    return res

def print_k_fold_avg_scores(train_scores_list, val_scores_list, gruop_scores_list=None):
    for dataset_name, score_lists in zip(['Train', 'Val'], [train_scores_list, val_scores_list]):
        avg_scores = defaultdict(float)
        counts = len(score_lists)

        for i, scores in enumerate(score_lists):
            logging.info(f"{dataset_name} flod {i} metric: {scores}")
            for score_type, score in scores.items():
                avg_scores[score_type] += score

        for score_type, total_score in avg_scores.items():
            logging.info(f"{dataset_name} Avg {score_type}: {total_score / counts}")
       
    if gruop_scores_list:
        group_scores = gruop_scores_list[0]
        
        for k in group_scores:
            avg_group_scores = {}
            # for group_scores in gruop_scores_list:
            for s_type in group_scores[k]:
                avg_group_scores[s_type] = sum([scores[k][s_type] for scores in gruop_scores_list if k in scores]) / len(gruop_scores_list)
            
            logging.info(f"{dataset_name} Avg Val {k}: {avg_group_scores}")
            