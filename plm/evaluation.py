import numpy as np


def get_f1(key, prediction, no_relation=0):
    """根据no_relation计算f1指标"""
    if no_relation == -1: # 没有no_relation就计算准确率
        # print('no-realtion == 1.')
        acc = (key == prediction).sum() / len(key)
        return acc, acc, acc
    correct_by_relation = ((key == prediction) & (prediction != no_relation)).astype(np.int32).sum()
    guessed_by_relation = (prediction != no_relation).astype(np.int32).sum()
    gold_by_relation = (key != no_relation).astype(np.int32).sum()

    prec_micro = 1.0
    if guessed_by_relation > 0:
        prec_micro = float(correct_by_relation) / float(guessed_by_relation)
    recall_micro = 1.0
    if gold_by_relation > 0:
        recall_micro = float(correct_by_relation) / float(gold_by_relation)
    f1_micro = 0.0
    if prec_micro + recall_micro > 0.0:
        f1_micro = 2.0 * prec_micro * recall_micro / (prec_micro + recall_micro)
    # print(prec_micro)
    # print(recall_micro)
    # print(f1_micro)
    return prec_micro, recall_micro, f1_micro
