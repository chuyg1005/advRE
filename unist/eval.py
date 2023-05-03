import os
from argparse import ArgumentParser

import torch
from data import RETACREDDataset, TACREDDataset
from eval_metric import tacred_f1, tacred_mi_ma_f1
from model import UniSTModel
from torch.utils.data import DataLoader, SequentialSampler
from tqdm import tqdm
from transformers import AutoTokenizer

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='../../re-datasets')
    parser.add_argument('--ckpt_dir', type=str, required=True)
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--split', type=str, required=True)
    parser.add_argument('--mask', action='store_true')

    args = parser.parse_args()
    # 读取参数
    ckpt_dir = args.ckpt_dir
    data_dir = args.data_dir
    dataset = args.dataset
    split = args.split
    data_path = os.path.join(data_dir, dataset, 'splits', split+'.json')
    no_task_desc = False
    mask_entity = args.mask
    model_name_or_path = 'roberta-base'
    eval_batch_size = 64
    max_sent_length=160
    max_label_length=20

    # 加载模型
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = UniSTModel.from_pretrained(ckpt_dir)
    model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(ckpt_dir)
    tokenizer.add_tokens(['<subj>', '<obj>'])
    model.roberta.resize_token_embeddings(len(tokenizer))

    # 加载数据集
    if dataset == 'tacred':
        Dataset =TACREDDataset
    elif dataset == 'retacred':
        Dataset = RETACREDDataset
    else:
        assert 0, f'no such dataset: {dataset}.'
    eval_dataset = Dataset(data_path, no_task_desc=no_task_desc, mode='test', mask_entity=mask_entity) 

    # 评价模型
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=eval_batch_size, collate_fn=lambda x: list(map(list, zip(*x))))
    
    dists = []
    labels = []
    model.eval()

    with torch.no_grad():
        labelset = eval_dataset.labelset
        labelset_inputs = tokenizer(labelset, padding=True, truncation=True, max_length=max_label_length, return_tensors="pt").to(device)
        labelset_embeddings = model.embed(**labelset_inputs)
    
    for sent, pos, *other in tqdm(eval_dataloader, desc="Evaluating"):
        labels.extend(pos)
        sent_inputs = tokenizer(sent, padding=True, truncation=True, max_length=max_sent_length, return_tensors="pt", is_split_into_words=True).to(device)
                         
        with torch.no_grad():            
            sent_embeddings = model.embed(**sent_inputs)
            for i in range(len(sent_embeddings)):
                sent_embedding = sent_embeddings[i].expand(labelset_embeddings.shape)
                dist = model.dist_fn(sent_embedding, labelset_embeddings).detach().cpu().numpy().tolist()
                dists.append(dist)  

    preds = []
    for i, dist in enumerate(dists):
        pred = min(labelset, key=lambda x:dist[labelset.index(x)])
        preds.append(pred)

    assert len(preds) == len(eval_dataset)
    # p, r, f1 = tacred_f1(labels, preds)
    # 计算micro_f1和macro_f1
    mi_f1, ma_f1 = tacred_mi_ma_f1(labels, preds, len(eval_dataset.labelset))

    # 打印结果
    # results = {
        # "Micro precision": p,
        # "Micro recall": r,
        # "Micro f1": f1
    # }

    print(split)
    print(f'mi_f1: {mi_f1*100:.2f}, ma_f1: {ma_f1*100:.2f}.')