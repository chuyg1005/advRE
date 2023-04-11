import json
import os
import random
from argparse import ArgumentParser

import numpy as np
import torch
# from constants import load_constants
from evaluation import get_f1
from model import REModel
from prepro import Processor
from transformers import AutoTokenizer
from utils import predict


def main(opt):
    # load data
    # rela2id, entity_type2id, entity_type_rela2id, entity_type_rela, subj_types, obj_types = load_constants(
# opt['data_dir'])
    with open(os.path.join(opt['data_dir'], 'rela2id.json')) as f:
        rela2id = json.load(f)
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
    prepro = Processor(opt['input_format'], tokenizer, opt['max_seq_length'], rela2id)
    eval_file = os.path.join(opt['eval_data_dir'], 'splits', opt['dataset'] + '.json')
    features = prepro.read(eval_file, mode='dev', mask_rate=opt['mask_rate'], all=opt['all'])

    # load model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = torch.load(os.path.join(opt['ckpt_dir'], f'best-{opt["model_name"]}.ckpt')).to(device)
    model.eval()

    eval_results_file = os.path.join(opt['ckpt_dir'], 'eval_results.json')
    eval_results = {}
    if os.path.exists(eval_results_file): eval_results = json.load(open(eval_results_file, 'r'))

    # predict results
    keys, preds = predict(model, features, opt['test_batch_size'], device)

    # compute f1 score and print
    keys, preds = np.array(keys, dtype=np.int64), np.array(preds, dtype=np.int64)
    _, _, f1 = get_f1(keys, preds, no_relation)  # 计算f1，并保存结果


    eval_results[opt['dataset']] = f1

    with open(eval_results_file, 'w') as f:
        json.dump(eval_results, f)

    id2rela = {id: rela for rela, id in rela2id.items()}
    key_relas = [id2rela[key] for key in keys]
    pred_relas = [id2rela[pred] for pred in preds]

    if opt['save']:
        save_path = os.path.join(opt['ckpt_dir'], opt['dataset']+'-pred.json')
        with open(save_path, 'w') as f:
            json.dump(list(zip(key_relas, pred_relas)), f)
    # filename = f'{args.dataset}_{args.input_format}.json'
    # with open(os.path.join(args.data_dir, filename), 'w') as f:
    #     json.dump(list(zip(key_relas, pred_relas)), f)

    print(f'{opt["dataset"]}: {f1 * 100:.2f}.')


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
