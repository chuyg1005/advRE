"""错误分析，验证baseline模型中预测错误的时候是将关系预测成为实体在训练集中的关系"""
import json
import os
import random
from argparse import ArgumentParser

import numpy as np
import torch
import torch.nn.functional as F
from constants import load_constants
from evaluation import get_f1
from model import REModel
from prepro import Processor
from torch import nn
from transformers import AutoTokenizer
from utils import collate_fn, predict


def main(opt):
    # 加载模型和prepro（预处理器）
    with open(os.path.join(opt['data_dir'], 'rela2id.json')) as f:
        rela2id = json.load(f)
    id2rela = {idx: rela for rela, idx in rela2id.items()}
    tokenizer = AutoTokenizer.from_pretrained(opt['model_name_or_path'])
    tokenizer_save_path = os.path.join(opt['data_dir'], opt['input_format'], 'tokenizer')
    if os.path.exists(tokenizer_save_path):
        print(f'load tokenizer from {tokenizer_save_path}.')
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_save_path)
    prepro = Processor(opt['input_format'], tokenizer, opt['max_seq_length'], rela2id)

    eval_file = os.path.join(opt['eval_data_dir'], 'splits', opt['dataset'] + '.json')
    train_file = os.path.join(opt['eval_data_dir'], 'train.json')
    train_data = json.load(open(train_file))
    dev_data = json.load(open(eval_file))
    entity_pair_dict = {}
    for item in train_data:
        tokens = item['token']
        ent_ss, ent_se, ent_os, ent_oe = item['subj_start'], item['subj_end'], item['obj_start'], item['obj_end']
        subj_name = ' '.join(tokens[ent_ss:ent_se+1])
        obj_name = ' '.join(tokens[ent_os:ent_oe+1])
        dict_key = subj_name + '&&' + obj_name
        if dict_key not in entity_pair_dict: entity_pair_dict[dict_key] = set()
        entity_pair_dict[dict_key].add(item['relation'])

    # 读取测试的数据集
    features = prepro.read(eval_file, mode='dev', mask_rate=opt['mask_rate'], all=opt['all'])

    # load model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = torch.load(os.path.join(opt['ckpt_dir'], f'{opt["model_name"]}.ckpt')).to(device)
    model.eval()

    keys, preds = predict(model, features, 64, device)
    keys = [id2rela[key] for key in keys]
    preds = [id2rela[pred] for pred in preds]

    total = 0
    cnt = 0

    for i in range(len(keys)):
        if keys[i] != preds[i]: # 做错了
            total += 1
            item = dev_data[i]
            tokens = item['token']
            ent_ss, ent_se, ent_os, ent_oe = item['subj_start'], item['subj_end'], item['obj_start'], item['obj_end']
            subj_name = ' '.join(tokens[ent_ss:ent_se+1])
            obj_name = ' '.join(tokens[ent_os:ent_oe+1])
            dict_key = subj_name + '&&' + obj_name
            if preds[i] in entity_pair_dict[dict_key] and preds[i] != 'no_relation': # 预测成了训练集中的关系，不包括NA
                cnt += 1

    print(f'total num: {len(keys)}, error num: {total}, entity-bias num: {cnt}.')


if __name__ == '__main__':
    parser = ArgumentParser()

    # parser.add_argument('--model_name_or_path', type=str, required=True)
    # parser.add_argument('--input_format', type=str, required=True)
    parser.add_argument('--ckpt_dir', type=str, required=True)
    # parser.add_argument('--data_dir', type=str, required=True, help='data dirname.')
    parser.add_argument('--dataset', type=str, required=True, help='eval dataset name.')

    parser.add_argument('--eval_data_dir', type=str, required=True, help='eval data directory.')
    parser.add_argument('--all', action='store_true', help='全句mask还是只是在实体部分mask')
    parser.add_argument('--mask_rate', type=float, default=0., help='实体名称遮蔽比例')
    parser.add_argument('--model_name',type=str, default='model', help='用来评价的模型名称')
    # parser.add_argument('--save', action='store_true', help='保存模型预测结果')
    # parser.add_argument('--data_dir', type=str, required=True, help='dataset dirname.')  # TODO: encoded or not encoded
    # parser.add_argument('--dataset', type=str, required=True, help='dataset name(without .json suffix).')
    # parser.add_argument('--kl_weight', type=float, required=False, help='kl div weight.') # 这一项并不重要
    # parser.add_argument('--max_seq_length', type=int, default=512)
    # parser.add_argument('--test_batch_size', type=int, default=128)

    args = parser.parse_args()

    config = json.load(open(os.path.join(args.ckpt_dir, 'configs.json'))) # 加载配置

    # opt = vars(args)
    # opt.update(config)
    opt = config
    opt.update(vars(args))
    random.seed(0)
    np.random.seed(0) # 固定随机数种子
    # opt.update(vars(args))

    main(opt)
