from collections import Counter


# ufet metric, mostly Borrowed from https://github.com/uwnlp/open_type
def f1(p, r):
    if r == 0.:
        return 0.
    return 2 * p * r / float(p + r)


def macro(labels, preds):
    true_and_prediction = list(zip(labels, preds))
    num_examples = len(true_and_prediction)
    p = 0.
    r = 0.
    pred_example_count = 0.
    pred_label_count = 0.
    gold_label_count = 0.
    for true_labels, predicted_labels in true_and_prediction:
        if predicted_labels:
            pred_example_count += 1
            pred_label_count += len(predicted_labels)
            per_p = 0 if len(predicted_labels) == 0 else len(set(predicted_labels).intersection(set(true_labels))) / float(len(predicted_labels))
            p += per_p
        if len(true_labels):
            gold_label_count += 1
            per_r = len(set(predicted_labels).intersection(set(true_labels))) / float(len(true_labels))
            r += per_r
    precision = p / pred_example_count if pred_example_count > 0 else 0
    recall = r / gold_label_count if gold_label_count > 0 else 0

    return precision, recall, f1(precision, recall)

def macro_fewshot(labels, preds, target_labels):
    true_and_prediction = list(zip(labels, preds))
    num_examples = len(true_and_prediction)
    p = 0.
    r = 0.
    pred_example_count = 0.
    pred_label_count = 0.
    gold_label_count = 0.
    for true_labels, predicted_labels in true_and_prediction:
        # set calculations to focus on target labels e.g. zero-shot labels, 1-5 shot labels
        true_labels = set(true_labels).intersection(set(target_labels))
        predicted_labels = set(predicted_labels).intersection(set(target_labels))
 
        if predicted_labels:
            pred_example_count += 1
            pred_label_count += len(predicted_labels)
            per_p = 0 if len(predicted_labels) == 0 else len(set(predicted_labels).intersection(set(true_labels))) / float(len(predicted_labels))
            p += per_p
        if len(true_labels):
            gold_label_count += 1
            per_r = len(set(predicted_labels).intersection(set(true_labels))) / float(len(true_labels))
            r += per_r
    precision = p / pred_example_count if pred_example_count > 0 else 0
    recall = r / gold_label_count if gold_label_count > 0 else 0

    return precision, recall, f1(precision, recall)




# tacred
def tacred_f1(labels, preds):
    correct_by_relation, guessed_by_relation, gold_by_relation = 0, 0, 0
    for i in range(len(preds)):
        if preds[i] != "no relation":
            guessed_by_relation += 1
            if preds[i] == labels[i]:
                correct_by_relation += 1
        if labels[i] != "no relation":
            gold_by_relation += 1

    prec_micro = 1.0
    if guessed_by_relation > 0:
        prec_micro = float(correct_by_relation) / float(guessed_by_relation)
    recall_micro = 1.0
    if gold_by_relation > 0:
        recall_micro = float(correct_by_relation) / float(gold_by_relation)
    f1_micro = 0.0
    if prec_micro + recall_micro > 0.0:
        f1_micro = 2.0 * prec_micro * recall_micro / (prec_micro + recall_micro)
    return prec_micro, recall_micro, f1_micro

def tacred_mi_ma_f1(label, output, rel_num):
    # 转换为字典
    # print(label)
    # print(output)

    relas = set(label + output) # 求出所有label
    rela2id = {rela:i for i, rela in enumerate(relas)}
    label = [rela2id[l] for l in label]
    output = [rela2id[l] for l in output]
    na_num = rela2id['no relation']
    
    correct_by_relation = Counter()
    guess_by_relation = Counter()
    gold_by_relation = Counter()

    for i in range(len(output)):
        guess = output[i]
        gold = label[i]

        if guess == na_num:
            guess = 0
        elif guess < na_num:
            guess += 1

        if gold == na_num:
            gold = 0
        elif gold < na_num:
            gold += 1

        if gold == 0 and guess == 0:
            continue
        if gold == 0 and guess != 0:
            guess_by_relation[guess] += 1
        if gold != 0 and guess == 0:
            gold_by_relation[gold] += 1
        if gold != 0 and guess != 0:
            guess_by_relation[guess] += 1
            gold_by_relation[gold] += 1
            if gold == guess:
                correct_by_relation[gold] += 1

    f1_by_relation = Counter()
    recall_by_relation = Counter()
    prec_by_relation = Counter()
    for i in range(1, rel_num):
        recall = 0
        if gold_by_relation[i] > 0:
            recall = correct_by_relation[i] / gold_by_relation[i]
        precision = 0
        if guess_by_relation[i] > 0:
            precision = correct_by_relation[i] / guess_by_relation[i]
        if recall + precision > 0:
            f1_by_relation[i] = 2 * recall * precision / (recall + precision)
        else:
            f1_by_relation[i] = 0.
        recall_by_relation[i] = recall
        prec_by_relation[i] = precision

    micro_f1 = 0
    if sum(guess_by_relation.values()) != 0 and sum(correct_by_relation.values()) != 0:
        recall = sum(correct_by_relation.values()) / sum(gold_by_relation.values())
        prec = sum(correct_by_relation.values()) / sum(guess_by_relation.values())
        micro_f1 = 2 * recall * prec / (recall + prec)

    macro_f1 = sum(f1_by_relation.values()) / len(f1_by_relation)

    return micro_f1, macro_f1