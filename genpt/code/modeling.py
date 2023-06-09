import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from arguments import get_args, get_embedding_layer, get_model_classes
from prompt_encoder import PromptEncoder
from torch.cuda.amp import autocast
from transformers import BartConfig, BartForConditionalGeneration


class Model(torch.nn.Module):

    def __init__(self, args, tokenizer=None):

        super().__init__()
        model_classes = get_model_classes()
        model_class = model_classes[args.model_type]

        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id
        self.args = args
        self.max_seq_length = args.max_seq_length
        self.model_type = args.model_type

        self.pseudo_token = args.pseudo_token
        self.pseudo_token_id = self.tokenizer.get_vocab()[self.pseudo_token]
        self.mask_token_id = self.tokenizer.mask_token_id

        self.model = model_class['model'].from_pretrained(
            args.model_name_or_path,
            return_dict=False,
            cache_dir=args.cache_dir if args.cache_dir else None)

        self.model.resize_token_embeddings(len(tokenizer))
        self.embeddings = get_embedding_layer(args, self.model)

        self.hidden_size = self.embeddings.embedding_dim
        self.spell_length = sum(args.prompt_lens)
        self.prompt_encoder = PromptEncoder(args.prompt_lens, self.hidden_size, self.tokenizer, args)
        self.prompt_encoder = self.prompt_encoder.cuda()

        # self.loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
        self.softmax = nn.Softmax(dim=-1)

        # for decoder
        self.vocab_size = len(self.tokenizer)
        self.max_ent_type_length = self.args.max_ent_type_length
        self.max_label_length = self.args.max_label_length
        self.max_generate_length = self.max_ent_type_length + self.max_label_length + 1

        self.emb_grad = {}

        self.mean = 0.
        self.var = 1.
        self.m = 0.99
        self.eps = 1e-12


    def embed_input(self, queries):
        bz = queries.shape[0]
        if self.spell_length == 0:
            raw_embeds = self.embeddings(queries)  # (bs, max_len, hidden_size)

        else:
            queries_for_embedding = queries.clone()
            queries_for_embedding[(queries == self.pseudo_token_id)] = self.tokenizer.unk_token_id
            raw_embeds = self.embeddings(queries_for_embedding)  # (bs, max_len, hidden_size)

            blocked_indices = torch.nonzero(queries == self.pseudo_token_id, as_tuple=False).reshape((bz, self.spell_length, 2))[:, :, 1]  # bz
            replace_embeds = self.prompt_encoder()

            for bidx in range(bz):
                for i in range(self.prompt_encoder.spell_length):
                    raw_embeds[bidx, blocked_indices[bidx, i], :] = replace_embeds[i, :]
        return raw_embeds

    def compute_loss(self, input_ids, attention_mask, target_ids, target_mask, target_labels, reduction='mean'):
        inputs_embeds = self.embed_input(input_ids)
        # * attention_mask：用来处理encoder的padding, 1-d
        # * decoder_attention_mask：用来处理decoder的padding，1-d
        outputs = self.model(inputs_embeds=inputs_embeds,
                            attention_mask=attention_mask,
                            decoder_input_ids=target_ids,
                            decoder_attention_mask=target_mask,
                            return_dict=True,
                            output_hidden_states=True
                            )

        logits = outputs[0]  # (bs, max_len, vocab_size)
        # loss = self.loss_fct(logits.view(-1, logits.size(-1)), target_labels.view(-1), reduction=reduction)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target_labels.view(-1), reduction=reduction, ignore_index=-100)
        if reduction == 'none':
            loss = loss.reshape(target_labels.shape) # 重新reshape回去
            loss = loss.sum(dim=1)
            target_lens = target_mask.sum(dim=1)
            loss = loss / target_lens # 重新归一化
            # if torch.any(loss.isnan()):
            #     print(target_lens)
            #     print(loss)
            #     assert 0, 'loss exists nan'
        return loss, logits
    
    # @torch.no_grad()
    def compute_ours_new_loss(self, input_ids, attention_mask, target_ids, target_mask, target_labels):
        sz = input_ids.size(0) // 2
        loss, logits = self.compute_loss(input_ids, attention_mask, target_ids, target_mask, target_labels, reduction='none')

        loss1, loss2 = loss.clone().detach().chunk(2)
        aug = loss2 - loss1 # 越大说明entity-bias越严重
        self.mean = self.m * self.mean + (1 - self.m) * aug.mean().item() # 进行滑动平均
        self.var = self.m * self.var + (1 - self.m) * aug.var().item() # 方差的滑动平均
        aug = (aug - self.mean) / np.sqrt(self.var + self.eps) # 标准化为标准正态分布
        # aug = torch.where(aug > 3, aug, torch.full_like(aug, -1e12)) # 保留损失增加量大于mean + 3sigma的
        # 排除nan
        aug = torch.where(torch.isnan(aug), torch.full_like(aug, -1e12), aug)
        org = torch.zeros_like(aug)
        weights = torch.stack([org, aug], 0) # 拼接起来
        weights = F.softmax(weights, 0).flatten() # 进行softmax转换为logits.

        return 2 * torch.dot(weights, loss) / sz, logits


    def compute_scores(self, input_ids, attention_mask, target_ids, target_mask, target_labels, ent_pos):
        """计算每个句子的实体偏差得分"""
        def backward_hook(module, gin, gout):
            self.emb_grad['grad'] = gout[0].clone().detach() # [batch_size, length, word_dim
        hook = self.embeddings.register_backward_hook(backward_hook)
        loss, logits = self.compute_loss(input_ids, attention_mask, target_ids, target_mask, target_labels)

        loss.backward()
        hook.remove()

        scores = self.emb_grad['grad'].norm(dim=2) # [batch_size, length]
        # print(scores)
        self.emb_grad = {} # 删除grad
        scores = scores / scores.sum(dim=1, keepdims=True) # 所有数值都是正数，归一化到【0，1】

        sent_lens = attention_mask.sum(dim=1)
        subj_lens = ent_pos[:, 1] - ent_pos[:, 0] + 1 + ent_pos[:, 3] - ent_pos[:, 2] + 1
        obj_lens = ent_pos[:, 5] - ent_pos[:, 4] + 1 + ent_pos[:, 7] - ent_pos[:, 6] + 1
        ent_lens = subj_lens + obj_lens
        baseline = ent_lens / sent_lens
        sent_scores = []
        for i in range(logits.size(0)):
            subj_score = scores[i][ent_pos[i, 0]:ent_pos[i,1]+1].sum() + \
                        scores[i][ent_pos[i,2]:ent_pos[i,3]+1].sum()
            obj_score = scores[i][ent_pos[i, 4]:ent_pos[i, 5] + 1].sum() + \
                        scores[i][ent_pos[i, 6]:ent_pos[i,7]+1].sum()
            sent_score = subj_score + obj_score
            sent_scores.append(sent_score)
        sent_scores = torch.tensor(sent_scores).to(baseline.device)
        # print(sent_scores.shape)
        # print(baseline.shape)
        sent_scores = sent_scores - baseline[:sent_scores.numel()]
        sent_scores = torch.clip(sent_scores, min=0).clone().detach()
        sent_scores = torch.where(torch.isnan(sent_scores), torch.zeros_like(sent_scores), sent_scores)

        self.zero_grad() # 清空梯度

        return sent_scores


    # @autocast() # 使用混合精度训练提高速度
    def forward(self, input_ids, attention_mask, target_labels, target_ids=None, target_mask=None, labels=None, ent_pos=None):
        # ent_pos: [batch_size, 4] / (ss, se, prpt_ss, prpt_se)
        sz = input_ids.size(0) // 2
        # 0表示baseline, 2表示直接混合伪数据训练，1表示使用我们的方法
        if self.args.train_mode == 'baseline':
            loss, logits = self.compute_loss(input_ids[:sz], attention_mask[:sz], target_ids[:sz], target_mask[:sz], target_labels[:sz])
            return loss, logits
        elif self.args.train_mode == 'data-aug':
            loss, logits = self.compute_loss(input_ids, attention_mask, target_ids, target_mask, target_labels)
            return loss, logits
        elif self.args.train_mode == 'ours':
            scores = self.compute_scores(input_ids[:sz], attention_mask[:sz], target_ids[:sz], target_mask[:sz], target_labels[:sz], ent_pos[:sz])
            loss, logits = self.compute_loss(input_ids, attention_mask, target_ids, target_mask, target_labels, reduction='none')

            loss1, loss2 = loss.chunk(2) # 划分为两部分

            return loss1.mean() + torch.dot(loss2, scores) / sz, logits
        elif self.args.train_mode == 'ours_new':
            return self.compute_ours_new_loss(input_ids, attention_mask, target_ids, target_mask, target_labels)



    @torch.no_grad()
    def greedy_decode(self, input_ids, attention_mask, ent_type_ids, ent_type_mask, type_pairs_index, labels=None):
        self.model.eval()
        inputs_embeds = self.embed_input(input_ids)

        batch_size = input_ids.size()[0]
        batch_index = torch.tensor([b for b in range(batch_size)]).cuda()
        tgt_logits = torch.zeros(batch_size, self.max_label_length + 1, self.vocab_size).cuda()

        decoder_input_ids = torch.tensor([self.pad_token_id]).unsqueeze(0).expand(batch_size, self.max_generate_length).contiguous().cuda()

        # entity type guided generation
        decoder_input_ids[:, 0] = 0
        decoder_input_ids[:, 1:ent_type_ids.shape[1] + 1] = ent_type_ids
        decoder_mask = torch.zeros_like(decoder_input_ids).cuda()
        decoder_mask[:, 0] = 1
        decoder_mask[:, 1:ent_type_ids.shape[1]+1] = 1

        decoder_index = torch.sum(ent_type_mask, dim=-1).long()  # (bs)
        index = torch.zeros_like(decoder_index)

        for i in range(self.max_label_length + 1):

            outputs = self.model(inputs_embeds=inputs_embeds,
                                 attention_mask=attention_mask,
                                 decoder_input_ids=decoder_input_ids,
                                 decoder_attention_mask=decoder_mask,
                                 )
            logits = outputs[0][(batch_index, decoder_index)]  # # (bs, vocab_size)

            tgt_logits[(batch_index, index)] = self.softmax(logits)
            topi = torch.argmax(logits, -1)
            decoder_index += 1
            index += 1
            if i < self.max_label_length:
                decoder_input_ids[(batch_index, decoder_index)] = topi
                decoder_mask[(batch_index, decoder_index)] = 1

        return tgt_logits


class RobertaModel(Model):

    def forward(self, input_ids, attention_mask, target_labels, target_ids=None, target_mask=None, labels=None):
        inputs_embeds = self.embed_input(input_ids)

        results = self.model(inputs_embeds=inputs_embeds,
                            attention_mask=attention_mask,
                            return_dict=True,
                            output_hidden_states=True,
                            )
        logits = results[0][:, self.max_seq_length:, :].contiguous()  # # (bs, tgt_length, vocab_size)
        loss = self.loss_fct(logits.view(-1, logits.size(-1)), target_labels.view(-1))

        return loss, logits


    @torch.no_grad()
    def greedy_decode(self, input_ids, attention_mask, ent_type_ids, ent_type_mask, type_pairs_index=None, labels=None):
        # 采用argmax解码，结果是[batch_sz, max_len, vacab_size]
        self.model.eval()
        batch_size = input_ids.size()[0]
        src_input_ids = input_ids[:, :self.max_seq_length]

        batch_index = torch.tensor([b for b in range(batch_size)]).cuda()
        tgt_logits = torch.zeros(batch_size, self.max_label_length + 1, self.vocab_size).cuda()
        decoder_input_ids = torch.tensor([self.pad_token_id]).unsqueeze(0).expand(batch_size, self.max_generate_length).contiguous().cuda()

        # entity type guided generation
        decoder_input_ids[:, 0] = 0
        decoder_input_ids[:, 1:ent_type_ids.shape[1]+1] = ent_type_ids

        decoder_index = torch.sum(ent_type_mask, dim=-1).long()  # (bs)
        index = torch.zeros_like(decoder_index)

        for i in range(self.max_label_length + 1):
            input_ids = torch.cat((src_input_ids, decoder_input_ids), dim=1)
            inputs_embeds = self.embed_input(input_ids)

            outputs = self.model(inputs_embeds=inputs_embeds,
                                 attention_mask=attention_mask
                                 )
            logits = outputs[0][(batch_index, decoder_index + self.max_seq_length)]  # (bs, vocab_size)
            tgt_logits[(batch_index, index)] = self.softmax(logits)
            topi = torch.argmax(logits, -1)
            decoder_index += 1
            index += 1
            if i < self.max_label_length:
                decoder_input_ids[(batch_index, decoder_index)] = topi

        return tgt_logits

def get_model(tokenizer):
    args = get_args()
    if args.model_type != 'roberta': # bart / t5
        model = Model(args, tokenizer)
    else:
        model = RobertaModel(args, tokenizer)
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    model.cuda()
    return model


def get_tokenizer(special=[]):
    args = get_args()
    model_classes = get_model_classes()
    model_config = model_classes[args.model_type]
    tokenizer = model_config['tokenizer'].from_pretrained(
        args.model_name_or_path,
        do_basic_tokenize=False,
        cache_dir=args.cache_dir if args.cache_dir else None, use_fast=True)
        # cache_dir=args.cache_dir if args.cache_dir else None) # 使用fast-tokenizer
    # tokenizer.add_tokens(special)
    tokenizer.add_special_tokens({'additional_special_tokens': special})
    return tokenizer
