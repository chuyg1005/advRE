import json
import os
import random
import sys
import time
from operator import itemgetter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from arguments import get_args_parser
from data_prompt import get_data
from modeling import get_model, get_tokenizer
from optimizing import get_optimizer
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm, trange
from utils import f1_score


def evaluate(model, val_dataset, val_dataloader, save_path=None):
    model.eval()
    scores = []
    all_labels = []
    NA_NUM = val_dataset.NA_NUM
    with torch.no_grad():
        for batch in tqdm(val_dataloader):
            labels = batch[-1].numpy().tolist()
            # 长度为6，6个元素
            batch = [item.cuda() for item in batch]
            type_pairs = batch[-2]
            logits = model.greedy_decode(*batch)
            res = []
            for bs in range(len(labels)): # 处理第bs个样本
                res_b = torch.zeros(len(val_dataset.prompt_id_2_label))
                logit = logits[bs]  # (max_len, vocab_size)
                for idx, i in enumerate(val_dataset.prompt_id_2_label):
                    _res = 0.0
                    for j in range(len(i)):
                        if i[j] != -100:
                            _res += logit[j, i[j]]  # (bs)
                    _res = _res / (len(i))
                    _res = _res.detach().cpu()
                    res_b[idx] = _res
                res_b = res_b * val_dataset.type_mapping[type_pairs[bs].item()] # 解码的时候会用到type_mapping的方式进行解码，0的直接盖住了
                res.append(res_b)
            logits = torch.stack(res, 0)  # (bs, max_rel)

            all_labels += labels
            scores.append(logits.cpu().detach())

        scores = torch.cat(scores, 0)
        scores = scores.detach().cpu().numpy()
        all_labels = np.array(all_labels)

        pred = np.argmax(scores, axis=-1).tolist()
        pred = np.array([NA_NUM if p == NA_NUM + 1 else p for p in pred])

        # 保存pred的结果

        mi_f1, ma_f1 = f1_score(pred, all_labels, val_dataset.num_class, val_dataset.NA_NUM)

        # 保存模型预测的结果
        if save_path:
            with open(save_path, 'w') as f:
                json.dump(list(zip(map(int, pred), map(int, all_labels))), f)

        return mi_f1, ma_f1

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(seed)

args = get_args_parser()
set_seed(args.seed)
tokenizer = get_tokenizer(special=[args.pseudo_token])
dadaset = get_data(args)

train_dataset = dadaset(
    path=args.data_dir,
    # name='train.json',
    name=args.train_name + ".json",
    rel2id=args.rel2id_dir,
    tokenizer=tokenizer,
    pseudo_token=args.pseudo_token,
    prompt_lens=args.prompt_lens,
    mode="train",
) if not args.eval_only else None

val_dataset = dadaset(
    path=args.data_dir,
    name='dev.json',
    rel2id=args.rel2id_dir,
    tokenizer=tokenizer,
    pseudo_token=args.pseudo_token,
    prompt_lens=args.prompt_lens,
    mode="dev"
) if not args.eval_only else None

unseen_dataset = dadaset(
    path=args.data_dir + "/splits",
    name='test_two_new_entity.json',
    rel2id=args.rel2id_dir,
    tokenizer=tokenizer,
    pseudo_token=args.pseudo_token,
    prompt_lens=args.prompt_lens,
    mode="dev"
) if not args.eval_only else None


challenge_dataset = dadaset(
    path=args.data_dir + "/splits",
    name='test_challenge.json',
    rel2id=args.rel2id_dir,
    tokenizer=tokenizer,
    pseudo_token=args.pseudo_token,
    prompt_lens=args.prompt_lens,
    mode="dev"
) if not args.eval_only else None


# 指向自定义划分之后的数据集路径
test_dataset = dadaset(
    path=args.data_dir+"/splits",
    # name='test.json',
    name=args.eval_name + '.json',
    rel2id=args.rel2id_dir,
    tokenizer=tokenizer,
    pseudo_token=args.pseudo_token,
    prompt_lens=args.prompt_lens,
    mode="test",
    mask_entity=args.mask_entity
)

train_batch_size = args.per_gpu_train_batch_size
val_batch_size = args.per_gpu_eval_batch_size

def collate_fn(batch, mode='train'):
    # print(len(batch)) # batch_size, 4
    # print(len(batch[0])) # batch的第一条数据, 7
    # if len(batch[0]) != 2: # 没有增强的数据
        # batch = list(map(torch.stack, map(list, zip(*batch)))) # 不使用外部数据
    # else: # 有增强的数据
    if mode == 'train': # 训练模式具有两条数据
        batch1 = [d[0] for d in batch]
        batch2 = [d[1] for d in batch]
        batch = batch1 + batch2
    # print(len(batch)) # 7
    # print(len(batch[0]))
    batch = list(map(torch.stack, map(list, zip(*batch))))

    return batch

# 训练模型的时候需要使用
if not args.eval_only:
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=train_batch_size, collate_fn=collate_fn)

    val_sampler = SequentialSampler(val_dataset)
    val_dataloader = DataLoader(val_dataset, sampler=val_sampler, batch_size=val_batch_size)

    unseen_sampler = SequentialSampler(unseen_dataset)
    unseen_dataloader = DataLoader(unseen_dataset, sampler=unseen_sampler, batch_size=val_batch_size)
    challenge_sampler = SequentialSampler(challenge_dataset)
    challenge_datalaoder = DataLoader(challenge_dataset, sampler=challenge_sampler, batch_size=val_batch_size)

# 测试模型
test_sampler = SequentialSampler(test_dataset)
test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=val_batch_size)

model = get_model(tokenizer)

data_name = args.data_name
model_type = args.model_type
if not args.eval_only:
    optimizer, scheduler, optimizer_new_token, scheduler_new_token = get_optimizer(model, train_dataloader)
    criterion = nn.CrossEntropyLoss()
    mx_res = 0.0
    hist_mi_f1 = []
    hist_ma_f1 = []
    path = args.output_dir + "/"
    os.makedirs(path, exist_ok=True)

if data_name == 'tacrev':
    data_name = 'tacred' # tacrev直接使用在tacred数据集上训练得到的模型

if args.k != 0 and args.data_seed != 0:
    checkpoint_prefix = '-'.join([str(model_type), str(data_name), str(args.k), str(args.data_seed)])
else:
    checkpoint_prefix = '-'.join([str(model_type), str(data_name)])
print(sys.argv)

# 训练模型
if not args.eval_only:
    start_train_time = time.time()
    for epoch in trange(int(args.num_train_epochs), desc="Epoch"):
        model.zero_grad()
        tr_loss = 0.0 # 历史loss的平均，
        global_step = 0

        total = len(train_dataloader)
        for step, batch in enumerate(train_dataloader):
            model.train()
            # 这边是7个数据
            # batch = [item.cuda() for item in batch]
            inputs = {
                "input_ids": batch[0].cuda(),
                "attention_mask": batch[1].cuda(),
                "target_labels": batch[2].cuda(),
                "target_ids": batch[3].cuda(),
                "target_mask": batch[4].cuda(),
                "labels": batch[5].cuda(),
                "ent_pos": batch[6].cuda()
            }
            loss, _ = model(**inputs) # logits并没有使用到，
            # print(f"loss: {loss.item()}.")

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()
            tr_loss += loss.item()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step() # 优化原始的模型参数
                scheduler.step()
                optimizer_new_token.step() # 优化新引入的token
                scheduler_new_token.step() 
                model.zero_grad()
                global_step += 1
                # print(tr_loss / global_step)
                sys.stdout.write('step: {} / {} | loss: {}\r'.format(step + 1, len(train_dataloader), tr_loss / global_step)) # \r回车不换行
                # print('step: {0:4} / {0:4} | loss: {1:2.6f}%'.format(step + 1, len(train_dataloader), tr_loss / global_step)) # \r回车不换行
                sys.stdout.flush()

        mi_f1, ma_f1 = evaluate(model, val_dataset, val_dataloader)
        chall_mi_f1, chall_ma_f1 = evaluate(model, challenge_dataset, challenge_datalaoder)
        un_mi_f1, un_ma_f1 = evaluate(model, unseen_dataset, unseen_dataloader)

        print("***** Epoch {} Validate *****: mi_f1 {}, ma_f1 {}".format(epoch, mi_f1, ma_f1))
        print("***** Epoch {} Challenge *****: mi_f1 {}, ma_f1 {}".format(epoch, chall_mi_f1, chall_ma_f1))
        print("***** Epoch {} Unseen *****: mi_f1 {}, ma_f1 {}".format(epoch, un_mi_f1, un_ma_f1))
        hist_mi_f1.append(mi_f1)
        hist_ma_f1.append(ma_f1)
        if mi_f1 > mx_res: # 保存最高的micro_f1_score的
            mx_res = mi_f1
            torch.save(model.state_dict(), args.output_dir + "/" + '{}-best_parameter'.format(checkpoint_prefix) + ".pkl")
        if epoch >= args.num_train_epochs // 2: # 保存后三个模型
            torch.save(model.state_dict(), args.output_dir + "/" + "{}-{}".format(checkpoint_prefix, epoch) + ".pkl") # 保存对应的checkpoint
        # if epoch == args.num_train_epochs - 1:
            # torch.save(model.state_dict(), args.output_dir + "/" + '{}-final_parameter'.format(checkpoint_prefix) + ".pkl")
    end_train_time = time.time()

    print(hist_mi_f1)
    # print(hist_ma_f1)
    print(mx_res)
    print("train time cost", end_train_time - start_train_time)

# print("***** Test on final model *****")
# start_test_time = time.time()
# mi_f1, ma_f1 = evaluate(model, test_dataset, test_dataloader)
# end_test_time = time.time()
# print("mi_f1 {}, ma_f1 {}".format(mi_f1, ma_f1))
# print(mi_f1)
# print("train time cost", end_train_time - start_train_time)
# print("test time cost", end_test_time - start_test_time)

# 加载模型
print("***** {} Test on best model *****".format(checkpoint_prefix))
model.load_state_dict(torch.load(args.output_dir + "/" + '{}-best_parameter'.format(checkpoint_prefix) + ".pkl"))
start_test_time = time.time()
if args.eval_only:
    # 保存到eval_name里面
    save_path = args.eval_name + '.json' if not args.mask_entity else args.eval_name + '_mask.json'
    mi_f1, ma_f1 = evaluate(model, test_dataset, test_dataloader, save_path=os.path.join(args.output_dir, save_path))
else:
    mi_f1, ma_f1 = evaluate(model, test_dataset, test_dataloader)
end_test_time = time.time()
print("mi_f1 {}, ma_f1 {}".format(mi_f1, ma_f1))
print("test time cost", end_test_time - start_test_time)
