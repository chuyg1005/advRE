import json
import os
from argparse import ArgumentParser

import numpy as np
import torch
from torch import nn
from constants import load_constants
from evaluation import get_f1
from model import REModel
from prepro import Processor
from transformers import AutoTokenizer
from utils import predict, collate_fn
import random
import torch.nn.functional as F


def main(opt):
    # load data
    rela2id, entity_type2id, entity_type_rela2id, entity_type_rela, subj_types, obj_types = load_constants(
opt['data_dir'])
    no_relation = -1
    if 'no_relation' in rela2id:
        no_relation = rela2id['no_relation']
    elif 'Other' in rela2id:
        no_relation = rela2id['Other']

    tokenizer = AutoTokenizer.from_pretrained(opt['model_name_or_path'])
    tokenizer_save_path = os.path.join(opt['data_dir'], opt['input_format'], 'tokenizer')
    if os.path.exists(tokenizer_save_path):
        print(f'load tokenizer from {tokenizer_save_path}.')
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_save_path)
    prepro = Processor(opt['input_format'], tokenizer, opt['max_seq_length'], subj_types, obj_types, rela2id,
                       entity_type2id)
    eval_file = os.path.join(opt['eval_data_dir'], 'splits', opt['dataset'] + '.json')
    # 输入一个句子
    data = json.load(open(eval_file))
    item1 = data[0]
    item2 = data[1]
    feature1 = prepro.get_feature(item1)
    feature2 = prepro.get_feature(item2)
    # features = [feature, feature] # 打包为batch的形式
    features = [feature1]
    # features = prepro.read(eval_file, mode='dev', mask_rate=opt['mask_rate'], all=opt['all'])

    # 输入是所有的features

    # load model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = torch.load(os.path.join(opt['ckpt_dir'], f'best-{opt["model_name"]}.ckpt')).to(device)
    model.eval()

    # 将features输入到模型
    batch = collate_fn(features)
    se = batch[7].to(device)
    oe = batch[8].to(device)
    # print(batch[0].shape)
    input_ids = batch[0].to(device)
    inputs = {'input_ids': input_ids,
                'attention_mask': batch[1].to(device),
                'ss': batch[3].to(device),
                'os': batch[4].to(device),
                'subj_type': batch[5].to(device),
                'obj_type': batch[6].to(device),
                # 'labels': batch[2].to(device)
                # 'mask_idx': batch[7].to(args.device)
            }
    emb_grad = {}
    def backward_hook(module, gin, gout):
        print('backward function is called.')
        emb_grad['grad'] = gout[0].clone().detach()
        # print(gin)
        # print(gout[0].shape)
    # model.retain_grad()
    # logit.retain_grad()
    # model.encoder.embeddings.word_embeddings.retain_grad()
    hook = model.encoder.embeddings.word_embeddings.register_full_backward_hook(backward_hook)

    logit = model(**inputs)
    # # 计算损失
    loss = F.cross_entropy(logit, batch[2].to(device))
    # model.loss_fnt.label_smoothing = 0.
    # loss_fnt = nn.CrossEntropyLoss()
    # loss = loss_fnt(logit, batch[2].to(device))
    # print(loss)
    # # for (name, module) in model.named_modules():
    #     print(name)

    loss.backward()

    # 词向量的梯度：[batch_size, length, word_dim]
    print(emb_grad['grad'][0, 0])
    print(model.encoder.embeddings.word_embeddings.weight.grad[input_ids[0, 0]])
    scores = emb_grad['grad'].norm(dim=2) # [batch_size, length]
    # 归一化到[0,1]
    # scores = (scores - scores.min(dim=1)[0]) / (scores.max(dim=1)[0] - scores.min(dim=1)[0])
    # print(scores.shape)
    scores = scores / scores.sum(dim=1, keepdims=True) # 所有数值都是正数，归一化到【0，1】
    # print(scores)
    sent_lens = batch[1].sum(dim=1)
    subj_lens = batch[7] - batch[3] + 1
    obj_lens = batch[8] - batch[4] + 1
    ent_lens = subj_lens + obj_lens # 计算实体的长度
    baseline = ent_lens / sent_lens  # 基准分数（整个句子的平均）
    # 已经计算出每个token的重要性，计算相对重要性来得到这个句子的得分
    sent_scores = []
    for i in range(scores.size(0)):
        ss, se, os_, oe = batch[3][i], batch[7][i], batch[4][i], batch[8][i]
        subj_score = scores[i][ss:se+1].sum()
        obj_score = scores[i][os_:oe+1].sum()
        sent_score = subj_score + obj_score
        print(sent_score, baseline[i])
        sent_score = sent_score - baseline[i].cuda()
        sent_scores.append(sent_score)
    sent_scores = torch.tensor(sent_scores)
    sent_scores = torch.clip(sent_scores, min=0) # 之后作为第二个batch的权重
    print(sent_scores)
    # print(emb_grad['grad'].shape)
    # print(dir(model.encoder.embeddings.word_embeddings))
    # print(model.encoder.embeddings.word_embeddings.weight.grad)

    hook.remove() 
    # print(loss)


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
    parser.add_argument('--save', action='store_true', help='保存模型预测结果')
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
