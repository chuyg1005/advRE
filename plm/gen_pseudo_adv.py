"""直接将实体名称替换为实体类型"""
import copy
import json
import os
import random
from argparse import ArgumentParser

import numpy as np
import torch
from prepro import Processor
from tqdm import tqdm
from transformers import AutoTokenizer
from utils import predict

"""
1. 统计实体字典
2. 进行随机替换
3. 保存新的数据
"""
parser = ArgumentParser()
parser.add_argument('--dataset', type=str, required=True, help='数据集名称.')
parser.add_argument('--input', type=str, default='train')
parser.add_argument('--output', type=str, default='train-adv')

args = parser.parse_args()

dataset = args.dataset
# 数据是train.json，输出是train-aug.json
data_dir = '../../re-datasets'
filepath = os.path.join(data_dir,dataset, args.input + '.json')
output = os.path.join(data_dir,dataset, args.output + '.json')

# 加载模型
ckpt_dir = os.path.join('saved_models', dataset, 'entity-name-only')
opt = json.load(open(os.path.join(ckpt_dir, 'configs.json'))) # 加载配置
with open(os.path.join(opt['data_dir'], 'rela2id.json')) as f:
    rela2id = json.load(f)
tokenizer = AutoTokenizer.from_pretrained(opt['model_name_or_path'])
tokenizer_save_path = os.path.join(opt['data_dir'], opt['input_format'], 'tokenizer')
if os.path.exists(tokenizer_save_path):
    print(f'load tokenizer from {tokenizer_save_path}.')
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_save_path)
prepro = Processor(opt['input_format'], tokenizer, opt['max_seq_length'], rela2id)
# load model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# 加载模型
model = torch.load(os.path.join(ckpt_dir, f'best-{opt["model_name"]}.ckpt')).to(device)
model.eval()



entity_dict = {}
data = json.load(open(filepath))

for d in tqdm(data):
    subj_type = d['subj_type']
    obj_type = d['obj_type']
    relation = d['relation']

    tokens = d['token']
    ss, se = d['subj_start'], d['subj_end']
    os_, oe = d['obj_start'], d['obj_end']

    subj_name = ' '.join(tokens[se:se+1])
    obj_name = ' '.join(tokens[os_:oe+1])

    if subj_type not in entity_dict: entity_dict[subj_type] = set()
    if obj_type not in entity_dict: entity_dict[obj_type] = set()

    entity_dict[subj_type].add(subj_name)
    entity_dict[obj_type].add(obj_name)

def substitute_entity(item, new_subj, new_obj):
    # 替换item中的subj和obj分别为new_subj和new_obj
    # new_item = copy.deepcopy(item)
    item = copy.deepcopy(item)
    ss, se = item['subj_start'], item['subj_end']
    os_, oe = item['obj_start'], item['obj_end']
    tokens = item['token']

    # subj位于obj之前
    if ss < os_:
        new_tokens = tokens[:ss] + new_subj
        new_ss, new_se = ss, len(new_tokens) - 1
        new_tokens += tokens[se+1:os_]
        new_os = len(new_tokens)
        new_tokens += new_obj
        new_oe = len(new_tokens) - 1
        new_tokens += tokens[oe+1:]
    else:
        new_tokens = tokens[:os_] + new_obj
        new_os, new_oe = os_, len(new_tokens) - 1
        new_tokens += tokens[oe+1:ss]
        new_ss = len(new_tokens)
        new_tokens += new_subj
        new_se = len(new_tokens) - 1
        new_tokens += tokens[se+1:] #* 应该加1，跳过se（重要！）
    
    item['token'] = new_tokens
    item['subj_start'] = new_ss
    item['subj_end'] = new_se
    item['obj_start'] = new_os
    item['obj_end'] = new_oe

    return item

random.seed(0)
# new_data = copy.deepcopy(data)
# 保留上下文不变替换实体
def generate_aug_data():

    # new_data = [[d] for d in data]
    keepN = 10 # 最终每个样本保留3条伪数据，每个样本传播10次选择困惑度最低的三条
    batch_size = 128
    new_data = []
    for d in tqdm(data):
        # org_d = copy.deepcopy(d)
        # d = copy.deepcopy(item[0]) 

        subj_type = d['subj_type']
        obj_type = d['obj_type']
        tokens = d['token']
        new_item = [d]
        candidates = []
        for _ in range(batch_size): # 额外生成keepN个样本
            # 转换为list
            if isinstance(entity_dict[subj_type], set):
                entity_dict[subj_type] = list(entity_dict[subj_type])
            if isinstance(entity_dict[obj_type], set):
                entity_dict[obj_type] = list(entity_dict[obj_type])
            new_subj = random.choice(entity_dict[subj_type]).split()
            new_obj = random.choice(entity_dict[obj_type]).split()

            candidate = substitute_entity(d, new_subj, new_obj)
            candidates.append(candidate)
        
        features = [prepro.get_feature(d) for d in candidates]
        keys, preds = predict(model, features, len(features), device)
        # 保留keys和preds不同的前keepN个结果
        # print(keys)
        # print(preds)
        indices = np.where(np.array(keys) != np.array(preds))[0]
        remains = np.where(np.array(keys) == np.array(preds))[0]
        # print(indices)

        new_item += [candidates[idx] for idx in indices][:keepN]
        # print(len(new_item))
        # 添加到keepN个
        if len(new_item) <= keepN:
            remain = keepN + 1 - len(new_item)
            new_item += [candidates[idx] for idx in remains][:remain]
        # if len(new_item) == 1: # 如果没有产生对抗样本，则使用自身代替
        #     print('no adv produce.')
        #     new_item += [d]
        new_data.append(new_item)
        # new_data.append(new_item)
        

    return new_data



new_data = generate_aug_data()
with open(output, 'w') as f: json.dump(new_data, f)