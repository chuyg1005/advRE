import copy
import json
import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0 and torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)


def convert_token(token):
    """ Convert PTB tokens to normal tokens """
    if (token.lower() == '-lrb-'):
        return '('
    elif (token.lower() == '-rrb-'):
        return ')'
    elif (token.lower() == '-lsb-'):
        return '['
    elif (token.lower() == '-rsb-'):
        return ']'
    elif (token.lower() == '-lcb-'):
        return '{'
    elif (token.lower() == '-rcb-'):
        return '}'
    return token


# def get_train_collate_fn(subj_dict, obj_dict, subs_rate=0.7):
#     def train_collate_fn(batch):
#         batch_size = len(batch)
#         # aug_batch = batch.copy()  # 数据增强的一批
#         # 将实体进行替换，然后增强
#         # 正则化的方式带来收益
#         aug_batch = copy.deepcopy(batch)  # 深拷贝
#         for i in range(batch_size):
#             rate1, rate2 = random.random(), random.random()
#             input_ids = batch[i]['input_ids']
#             ss, se, os, oe = batch[i]['ss'], batch[i]['se'], batch[i]['os'], batch[i]['oe']
#             subj_type, obj_type = batch[i]['subj_type'], batch[i]['obj_type']
#             old_subj, old_obj = input_ids[ss:se + 1], input_ids[os:oe + 1]
#             new_subj, new_obj = random.choice(subj_dict[str(subj_type)]), random.choice(obj_dict[str(obj_type)])

#             if ss < os:  # 头实体在尾实体的前面
#                 if rate1 < subs_rate:  # 替换头实体
#                     input_ids = input_ids[:ss] + new_subj + input_ids[se + 1:]
#                     se = se - len(old_subj) + len(new_subj)
#                     os = os - len(old_subj) + len(new_subj)
#                     oe = oe - len(old_subj) + len(new_subj)
#                 if rate2 < subs_rate:  # 替换尾实体
#                     input_ids = input_ids[:os] + new_obj + input_ids[oe + 1:]
#                     oe = oe - len(old_obj) + len(new_obj)

#             else:  # 尾实体在头实体的前面
#                 if rate1 < subs_rate:  # 替换尾实体
#                     input_ids = input_ids[:os] + new_obj + input_ids[oe + 1:]
#                     oe = oe - len(old_obj) + len(new_obj)
#                     ss = ss - len(old_obj) + len(new_obj)
#                     se = se - len(old_obj) + len(new_obj)
#                 if rate2 < subs_rate:  # 替换头
#                     input_ids = input_ids[:ss] + new_subj + input_ids[se + 1:]
#                     se = se - len(old_subj) + len(new_subj)
#             aug_batch[i]['input_ids'] = input_ids
#             aug_batch[i]['ss'], aug_batch[i]['se'] = ss, se
#             aug_batch[i]['os'], aug_batch[i]['oe'] = os, oe

#         # 先是一批正常的数据，然后是一批增强的数据
#         return collate_fn(batch + aug_batch)

#     return train_collate_fn


def collate_fn(batch, use_pseudo):
    # 需要处理一个元素具有两个的情况，如果是一个就是下面的流程，两个则需要额外处理一下
    # 检查一个元素是否是具有两个
    if use_pseudo:
        # 使用伪数据
        batch1 = [f[0] for f in batch]
        indices = np.random.randint(1, len(batch[0]), len(batch))
        batch2 = [batch[i][indices[i]] for i in range(len(batch))]
        batch = batch1 + batch2
    # if len(batch[0]) != 2:
    #     pass
    # else:
    #     batch1 = [f[0] for f in batch] # 原始的数据
    #     batch2 = [f[1] for f in batch] # 增强的数据
    #     batch = batch1 + batch2 # 将两个batch拼接起来
    max_len = max([len(f["input_ids"]) for f in batch])
    input_ids = [f["input_ids"] + [0] * (max_len - len(f["input_ids"])) for f in batch]
    input_mask = [[1.0] * len(f["input_ids"]) + [0.0] * (max_len - len(f["input_ids"])) for f in batch]
    labels = [f["labels"] for f in batch]
    se = [f['se'] for f in batch]
    ss = [f["ss"] for f in batch]
    os = [f["os"] for f in batch]
    oe = [f['oe'] for f in batch]

    # subj_type = [f['subj_type'] for f in batch]
    # obj_type = [f['obj_type'] for f in batch]
    # mask_idx = [f['mask_idx'] for f in batch]
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    input_mask = torch.tensor(input_mask, dtype=torch.float)
    labels = torch.tensor(labels, dtype=torch.long)
    ss = torch.tensor(ss, dtype=torch.long)
    os = torch.tensor(os, dtype=torch.long)
    se = torch.tensor(se, dtype=torch.long)
    oe = torch.tensor(oe, dtype=torch.long)

    # subj_type = torch.tensor(subj_type, dtype=torch.long)
    # obj_type = torch.tensor(obj_type, dtype=torch.long)
    # mask_idx = torch.tensor(mask_idx, dtype=torch.long)
    # output = (input_ids, input_mask, labels, ss, os, subj_type, obj_type, se, oe)  # , mask_idx)
    output = (input_ids, input_mask, labels, ss, se, os, oe)
    return output


def predict(model, features, test_batch_size, device):
    dataloader = DataLoader(features, batch_size=test_batch_size, collate_fn=collate_fn, drop_last=False)
    keys, preds = [], []
    model.eval()
    for i_b, batch in enumerate(tqdm(dataloader)):

        inputs = {'input_ids': batch[0].to(device),
                  'attention_mask': batch[1].to(device),
                  'ss': batch[3].to(device),
                #   'se': batch[4].to(device),
                  'os': batch[5].to(device),
                #   'oe': batch[6].to(device)
                #   'subj_type': batch[5].to(device),
                #   'obj_type': batch[6].to(device),
                  # 'mask_idx': batch[7].to(args.device)
                  }
        keys += batch[2].tolist()
        with torch.no_grad():
            logit = model(**inputs)  # 预测结果
            pred = torch.argmax(logit, dim=-1) # 
        preds += pred.tolist()
    return keys, preds


if __name__ == '__main__':
    data_dir = 'data/tacred/entity_mask_new'
    data = json.load(open(os.path.join(data_dir, 'train.json')))

    # train_collate_fn = get_train_collate_fn(subj_dict, obj_dict)
    # batch_data = data[:2]
    # tokenizer = AutoTokenizer.from_pretrained('roberta-base')

    # new_batch_data = collate_fn(batch_data)
    train_dataloader = DataLoader(data, batch_size=64, shuffle=True,
                                  collate_fn=collate_fn,
                                  drop_last=True)
    print(len(train_dataloader))

    # for i in range(2):
    #     print(tokenizer.decode(batch_data[i][0]))
    #     print(tokenizer.decode(batch_data[i+2][0]))
    #     print('=' * 100)
