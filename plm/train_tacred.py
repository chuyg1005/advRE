import argparse
import json
import os

import numpy as np
import torch
import ujson
# from constants import load_constants
from evaluation import get_f1
from model import REModel
# from prepro import TACREDProcessor
from prepro import Processor
from torch.cuda.amp import GradScaler
from torch.optim import AdamW
from torch.utils.data import DataLoader
# import wandb
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer
# from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from transformers.optimization import get_linear_schedule_with_warmup
from utils import collate_fn, predict, set_seed


def train(args, model, train_features, benchmarks, no_relation, ckpt_dir, eval_first=False, writer=None):
    # print(len(train_features))
    train_dataloader = DataLoader(train_features, batch_size=args.train_batch_size, shuffle=True,
                                  collate_fn=lambda batch: collate_fn(batch, 'train'),
                                  drop_last=False)
                                #   drop_last=True)
    total_steps = int(len(train_dataloader) * args.num_train_epochs // args.gradient_accumulation_steps)
    warmup_steps = int(total_steps * args.warmup_ratio)

    scaler = GradScaler()
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                num_training_steps=total_steps)
    print('Total steps: {}'.format(total_steps))
    print('Warmup steps: {}'.format(warmup_steps))

    num_steps = 0
    if not eval_first:
        best_dev_f1 = 0.
        test_f1 = 0.
    else:
        eval_results = eval_model(args, model, benchmarks, no_relation, num_steps, writer)
        best_dev_f1 = eval_results['dev']
        test_f1 = eval_results['test']
    evaluation_steps = len(train_dataloader) // 2  # 刚开始的时候一个epoch评价一次
    min_eval_step = 100
    # model.save(os.path.join(ckpt_dir, f'best-{args.model_name}.ckpt'))
    for epoch in range(int(args.num_train_epochs)):
        model.zero_grad()
        for step, batch in enumerate(tqdm(train_dataloader)):
            model.train()
            inputs = {'input_ids': batch[0].to(args.device),
                      'attention_mask': batch[1].to(args.device),
                      'labels': batch[2].to(args.device),
                      'ss': batch[3].to(args.device),
                      'se': batch[4].to(args.device),
                        'os': batch[5].to(args.device),
                        'oe': batch[6].to(args.device),
                        'train_mode': args.train_mode
                      }
            # outputs = model(**inputs)
            # # loss = outputs[0] / args.gradient_accumulation_steps
            # loss = outputs[0]
            # loss.backward() # 反向传播
            loss = model.compute_loss(**inputs) #TODO: 模型反向传播，包装了loss.backward（因为需要自己定制一下）；反向传播
            scaler.scale(loss).backward()
            # TODO: graident_accumulation_steps总是1
            # if step % args.gradient_accumulation_steps == 0:
            num_steps += 1
            if args.max_grad_norm > 0:
                # print('so large.')
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            model.zero_grad()
            # wandb.log({'loss': loss.item()}, step=num_steps)
            writer.add_scalar('train_loss', loss.item(), global_step=num_steps)

            # if (num_steps % args.evaluation_steps == 0 and step % args.gradient_accumulation_steps == 0):
            # 避免刚开始就进行评价
            if (num_steps + 1) % evaluation_steps == 0:
                # 训练结束一个epoch评价一次
                # eval_results = {}
                # for tag, features in benchmarks:
                #     f1, output = evaluate(args, model, features, tag=tag, no_relation=no_relation)
                #     wandb.log(output, step=num_steps)
                #     eval_results[tag] = f1
                eval_results = eval_model(args, model, benchmarks, no_relation, num_steps, writer)

                if eval_results['dev'] > best_dev_f1:
                    best_dev_f1 = eval_results['dev']
                    test_f1 = eval_results['test']
                    model.save(os.path.join(ckpt_dir, f'best-{args.model_name}.ckpt'))
                    # torch.save(model, os.path.join(ckpt_dir, f'best-{args.model_name}.ckpt'))
                # else:
                    # evaluation_steps = max(min_eval_step, evaluation_steps // 2)  # 缩短评价
        if epoch >= int(args.num_train_epochs) // 2: # 保存最后的几个模型
            model.save(os.path.join(ckpt_dir, f'checkpoint-{epoch}.ckpt'))

    print(f'best_dev_f1: {best_dev_f1}, test_f1: {test_f1}.')
    # for tag, features in benchmarks:
    #     f1, output = evaluate(args, model, features, tag=tag, no_relation=no_relation)
    #     wandb.log(output, step=num_steps)


def eval_model(args, model, benchmarks, no_relation, num_steps, writer):
    eval_results = {}
    for tag, features in benchmarks:
        f1, output = evaluate(args, model, features, tag=tag, no_relation=no_relation)
        # wandb.log(output, step=num_steps)
        writer.add_scalar(tag, f1, global_step=num_steps)
        # print(output)
        eval_results[tag] = f1
    return eval_results


def evaluate(args, model, features, tag='dev', no_relation=0):
    keys, preds = predict(model, features, args.test_batch_size, args.device)
    # dataloader = DataLoader(features, batch_size=args.test_batch_size, collate_fn=collate_fn, drop_last=False)
    # keys, preds = [], []
    # for i_b, batch in enumerate(tqdm(dataloader)):
    #     model.eval()
    #
    #     inputs = {'input_ids': batch[0].to(args.device),
    #               'attention_mask': batch[1].to(args.device),
    #               'ss': batch[3].to(args.device),
    #               'os': batch[4].to(args.device),
    #               'subj_type': batch[5].to(args.device),
    #               'obj_type': batch[6].to(args.device),
    #               # 'mask_idx': batch[7].to(args.device)
    #               }
    #     keys += batch[2].tolist()
    #     with torch.no_grad():
    #         logit = model(**inputs)[0]
    #         pred = torch.argmax(logit, dim=-1)
    #     preds += pred.tolist()

    keys = np.array(keys, dtype=np.int64)
    preds = np.array(preds, dtype=np.int64)
    _, _, max_f1 = get_f1(keys, preds, no_relation)

    output = {
        tag + "_f1": max_f1 * 100,
    }
    print(output)
    return max_f1, output


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", default="./data/tacred", type=str)
    parser.add_argument('--data_cache_dir', default='./data', type=str)
    parser.add_argument("--model_name_or_path", default="bert-base", type=str)
    parser.add_argument("--input_format", default="typed_entity_marker_punct", type=str,
                        help="in [entity_mask, entity_marker, entity_marker_punct, typed_entity_marker, typed_entity_marker_punct]")

    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--max_seq_length", default=512, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated.")

    parser.add_argument("--train_batch_size", default=128, type=int,
                        help="Batch size for training.")
    parser.add_argument("--test_batch_size", default=128, type=int,
                        help="Batch size for testing.")
    parser.add_argument("--learning_rate", default=3e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int,
                        help="Number of updates steps to accumulate the gradients for, before performing a backward/update pass.")
    parser.add_argument("--adam_epsilon", default=1e-6, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--warmup_ratio", default=0.1, type=float,
                        help="Warm up ratio for Adam.")
    parser.add_argument("--num_train_epochs", default=5.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--seed", type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument("--num_class", type=int, default=42)
    parser.add_argument("--evaluation_steps", type=int, default=500,
                        help="Number of steps to evaluate the model")

    parser.add_argument("--dropout_prob", type=float, default=0.1)
    parser.add_argument("--project_name", type=str, default="RE_baseline")
    parser.add_argument("--run_name", type=str, default="tacred")
    # 遮蔽不可能的关系，从而让模型有更好的表现
    parser.add_argument('--eval_first', action='store_true', help='先进行评价')
    parser.add_argument('--mask_rela', action='store_true', help='mask unknown relas.')
    parser.add_argument('--ckpt_dir', type=str, default='saved_models')
    parser.add_argument('--from_checkpoint', action='store_true')
    # parser.add_argument('--use_baseline', action='store_true') # 是否使用基础模型
    # parser.add_argument('--ablation', action='store_true')
    # parser.add_argument('--subs_rate', type=float, default=0., help='entity substitute rate.')
    # parser.add_argument('--kl_weight', type=float, default=0., help='kl div weight(0 means no kl div).')
    # parser.add_argument('--aug_weight', type=float, default=0., help='增加的伪数据的权重')
    parser.add_argument('--train_mode', type=str, default='baseline', help='baseline / data-aug / ours')
    parser.add_argument('--train_name', type=str, default='train', help='训练集的名字，用于后期扩展伪数据')
    parser.add_argument('--model_name', type=str, default='model', help='存储最优模型的名称（best-mdoel_name.ckpt）')

    args = parser.parse_args()
    # wandb.init(project=args.project_name, name=args.run_name)
    # 使用tensorboard替代wandb
    dataset = os.path.basename(args.data_dir)
    writer = SummaryWriter(os.path.join(args.project_name, dataset, args.run_name))

    ckpt_dir = os.path.join(args.ckpt_dir, dataset, args.run_name) # 保存ckpt的路径
    os.makedirs(ckpt_dir, exist_ok=True)  # 保存断点的路径

    with open(os.path.join(ckpt_dir, 'configs.json'), 'w') as f:
        json.dump(vars(args), f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device
    if args.seed > 0:  # 固定随机数种子
        set_seed(args)

    # config = AutoConfig.from_pretrained(
    #     args.config_name if args.config_name else args.model_name_or_path,
    #     num_labels=args.num_class,
    # )
    # config.gradient_checkpointing = True
    # tokenizer = AutoTokenizer.from_pretrained(
    #     args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
    # )

    # rela2id, entity_type2id, entity_type_rela2id, entity_type_rela, subj_types, obj_types = load_constants(
    #     args.data_dir) # 加载constant
    # print(os.path.join(args.data_dir, "rela2id.json"))
    with open(os.path.join(args.data_dir, 'rela2id.json')) as f:
        rela2id = json.load(f)
        print(f"数据集{args.data_dir}的关系数量为：{len(rela2id)}.")

    args.num_class = len(rela2id)  # TODO: 重新设置设置类别数量
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model = REModel(args.model_name_or_path, args.num_class, args.dropout_prob)
    model.to(args.device)

    # train_file = os.path.join(args.data_dir, "train.json")
    train_file = os.path.join(args.data_dir, args.train_name + '.json')
    dev_file = os.path.join(args.data_dir, "dev.json")
    test_file = os.path.join(args.data_dir, "test.json")
    # dev_rev_file = os.path.join(args.data_dir, "dev_rev.json")
    # test_rev_file = os.path.join(args.data_dir, "test_rev.json")

                        #   entity_type2id)
    processor = Processor(args.input_format, tokenizer, args.max_seq_length, rela2id) # 非use_baseline就是use_pseudo

    # 缓存一些生成的文件
    data_cache_dir = os.path.join(args.data_cache_dir, dataset, args.input_format)
    if not os.path.exists(data_cache_dir): os.makedirs(data_cache_dir)

    train_cache_path = os.path.join(data_cache_dir, args.train_name + '.json')
    dev_cache_path = os.path.join(data_cache_dir, 'dev.json')
    test_cache_path = os.path.join(data_cache_dir, 'test.json')


    def load_features(file_in, mode, cache_path):
        if not os.path.exists(cache_path):
            features = processor.read(file_in, mode=mode)
            with open(cache_path, 'w') as f:
                json.dump(features, f)
        else:
            features = json.load(open(cache_path))
        return features

    # 读取转换好的特征
    train_features = load_features(train_file, 'train', train_cache_path)
    dev_features = load_features(dev_file, 'dev', dev_cache_path)
    test_features = load_features(test_file, 'test', test_cache_path)

    # 保存tokenizer
    tokenizer_save_path = os.path.join(data_cache_dir, 'tokenizer')
    if not os.path.exists(tokenizer_save_path):
        tokenizer.save_pretrained(tokenizer_save_path)

    # no_relation = rela2id['no_relation'] # 用来计算f1指标的时候使用
    no_relation = -1 # 记录no-relation的标记
    if 'no_relation' in rela2id: no_relation = rela2id['no_relation']
    elif 'Other' in rela2id: no_relation = rela2id['Other']

    if len(processor.text_encoder.new_tokens) > 0:
        model.encoder.resize_token_embeddings(len(tokenizer))

    if args.from_checkpoint:
        model = torch.load(os.path.join(ckpt_dir, f'best-{args.model_name}.ckpt'))
        # model.entity_type_rela2id = entity_type_rela2id

    benchmarks = (
        ("dev", dev_features),
        ("test", test_features),
        # ("dev_rev", dev_rev_features),
        # ("test_rev", test_rev_features),
    )

    # 开始训练
    train(args, model, train_features, benchmarks, no_relation, ckpt_dir,
          eval_first=args.eval_first, writer=writer)


if __name__ == "__main__":
    main()
