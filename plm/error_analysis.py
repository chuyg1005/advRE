"""错误分析，验证baseline模型中预测错误的时候是将关系预测成为实体在训练集中的关系"""
import copy
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
from torch.utils.data import DataLoader
from tqdm import tqdm
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
    dev_data = json.load(open(eval_file))

    # load model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = torch.load(os.path.join(opt['ckpt_dir'], f'{opt["model_name"]}.ckpt')).to(device)
    model.eval()
    # keep = opt['hit']

    # rela_id = opt['rela_id']
    def analysis(rela_id, max_hit):
        rela = id2rela[rela_id]
        print(f'analysis for relation: {rela}. hit@{max_hit}.')

        # 准备数据
        features = []
        total = 0
        for d in tqdm(dev_data):
            if d['relation'] != rela: continue # 过滤掉不是目标关系类型的
            total += 1
            entity_only = copy.deepcopy(d)  # 只保留实体名称
            context_only = copy.deepcopy(d) # 只保留上下文
            tokens = d['token']
            ent_ss, ent_se, ent_os, ent_oe = d['subj_start'], d['subj_end'], d['obj_start'], d['obj_end']
            subj = tokens[ent_ss:ent_se+1] # head entity
            obj = tokens[ent_os:ent_oe+1] # tail entity
            entity_only['token'] = subj + ['and'] + obj
            entity_only['subj_start'] = 0
            entity_only['subj_end'] = len(subj) - 1
            entity_only['obj_start'] = len(subj) + 1
            entity_only['obj_end'] = len(subj) + len(obj)

            feature = prepro.get_feature(d, mask_rate=0., all=False)
            feature_e = prepro.get_feature(entity_only, mask_rate=0., all=False)
            feature_c = prepro.get_feature(context_only, mask_rate=1., all=False)
            # features += [feature1, feature2] # 每个原始样本对应两个特征
            features += [feature, feature_e, feature_c]



        # 开始预测，保证batch_size是3的倍数
        batch_size = 64
        dataloader = DataLoader(features, batch_size=batch_size * 3, collate_fn=lambda batch: collate_fn(batch, 'eval'), drop_last=False, shuffle=False) # 评价模型
        # keys, preds = [], []
        ent_hits = [0] * max_hit
        cont_hits = [0] * max_hit
        # 计算mrr指标
        for i_b, batch in enumerate(tqdm(dataloader)):

            inputs = {'input_ids': batch[0].to(device),
                    'attention_mask': batch[1].to(device),
                    'ss': batch[3].to(device),
                    'os': batch[5].to(device),
                    }
            with torch.no_grad():
                logits = model(**inputs)  # 预测结果
                logit = logits[::3]
                logit_e = logits[1::3] # 只使用实体名称的预测结果
                logit_c = logits[2::3] # 只使用上下文的预测结果

                pred = torch.argmax(logit, dim=1).unsqueeze(-1) # [batch_size, 1]
                for keep in range(1, max_hit+1):
                    _, logit_e_topk = logit_e.topk(keep ,dim=1)
                    _, logit_c_topk= logit_c.topk(keep, dim=1)

                    ent_hit = torch.any(pred == logit_e_topk, dim=1).sum()
                    cont_hit = torch.any(pred == logit_c_topk, dim=1).sum()

                    ent_hits[keep-1] += ent_hit
                    cont_hits[keep-1] += cont_hit
                # logit1, logit2 = logit.chunk(2) # logit1为原始的预测值，logit2为新的预测值
                # pred_org = torch.argmax(logit1, dim=-1)
                # rank = logit2.argsort(dim=1, descending=True) # 最大的是0
                # order = torch.gather(rank, 1, pred_org.unsqueeze(1))
                # sz = logit1.size(0)
                # rate = 1 / (order+1)
                # rates.append(rate.mean().item()) 
        
        # 计算出的ent_hit@k和cont_hit@k指标
        return total, ent_hits, cont_hits
        # print(f'ent_hit@{keep}: {ent_hits / total}; cont_hit@{keep}: {cont_hits / total}.')

    results = []
    max_hit = 5
    # for hit in range(1, 5):
    #     result = []
    for rela_id in id2rela:
        total, ent_hits, cont_hits = analysis(rela_id, max_hit)
        print(f'total: {total}, ent_hits: {ent_hits}, cont_hits: {cont_hits}.')
        results.append([id2rela[rela_id], total, ent_hits, cont_hits])

    with open('results1.json', 'w') as f:
        json.dump(results, f)

if __name__ == '__main__':
    parser = ArgumentParser()

    # parser.add_argument('--model_name_or_path', type=str, required=True)
    # parser.add_argument('--input_format', type=str, required=True)
    parser.add_argument('--ckpt_dir', type=str, required=True)
    # parser.add_argument('--data_dir', type=str, required=True, help='data dirname.')
    parser.add_argument('--dataset', type=str, required=True, help='eval dataset name.')

    parser.add_argument('--eval_data_dir', type=str, required=True, help='eval data directory.')
    # parser.add_argument('--all', action='store_true', help='全句mask还是只是在实体部分mask')
    # parser.add_argument('--mask_rate', type=float, default=0., help='实体名称遮蔽比例')
    parser.add_argument('--model_name',type=str, default='best-model', help='用来评价的模型名称')
    # parser.add_argument('--rela_id', type=int, default=0)
    # parser.add_argument('--hit', type=int, default=1)
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
