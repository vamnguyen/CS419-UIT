# evaluator.py
import math
import numpy as np

def precision_recall_f1(y_true_set, y_pred_list, k=None):
    if k is None:
        k = len(y_pred_list)
    pred_k = y_pred_list[:k]
    tp = sum(1 for d in pred_k if d in y_true_set)
    precision = tp / len(pred_k) if len(pred_k)>0 else 0.0
    recall = tp / len(y_true_set) if len(y_true_set)>0 else 0.0
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2*precision*recall/(precision+recall)
    return precision, recall, f1

def apk(actual, predicted, k):
    if len(predicted) > k:
        predicted = predicted[:k]
    score = 0.0
    num_hits = 0.0
    for i, p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i+1.0)
    if not actual:
        return 0.0
    return score / min(len(actual), k)

def mapk(actuals, predicted_lists, k):
    return np.mean([apk(a, p, k) for a,p in zip(actuals, predicted_lists)]) if actuals else 0.0

def dcg_at_k(rels, k):
    dcg = 0.0
    for i in range(k):
        if i < len(rels):
            dcg += (2**rels[i] - 1) / math.log2(i+2)
    return dcg

def ndcg_at_k(actual_set, predicted_list, k):
    rels = [1 if doc in actual_set else 0 for doc in predicted_list[:k]]
    dcg = dcg_at_k(rels, k)
    ideal_rels = sorted(rels, reverse=True)
    idcg = dcg_at_k(ideal_rels, k)
    return dcg/idcg if idcg>0 else 0.0
