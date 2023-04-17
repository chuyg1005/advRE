import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
from transformers import RobertaModel, RobertaPreTrainedModel


class UniSTModel(RobertaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)       
        self.roberta = RobertaModel(config)      
        # 启用梯度检查点会减少显存占用，但是会增加模型的训练时间（训练时间会翻倍，显存占用会降低）
        # * 显存占用从28G->8G；训练时间从5min -> 6min30s
        # * base模型没必要启用，large模型可以启用
        # self.roberta.gradient_checkpointing_enable()  # 启用梯度检查点
        self.margin = config.margin       
        self.init_weights()
        self.config = config
        self.emb_grad = {}

    def compute_loss(self, sent_embeddings, pos_embeddings, neg_embeddings, reduction='mean'):
        loss = F.triplet_margin_with_distance_loss(sent_embeddings, pos_embeddings, neg_embeddings, distance_function=self.dist_fn, margin=self.margin, reduction=reduction)
        return loss
        
    @autocast()
    def forward(
        self, 
        sent_input_ids,
        pos_input_ids,
        neg_input_ids,
        sent_attention_mask=None,
        pos_attention_mask=None,
        neg_attention_mask=None,
        ss=None,
        se=None,
        os=None,
        oe=None,
        desc_ss=None,
        desc_se=None,
        desc_os=None,
        desc_oe=None
    ):
        # 不使用伪数据
        if not self.config.use_pseudo or self.config.ablation:
            # with autocast():
            sent_embeddings = self.embed(sent_input_ids, sent_attention_mask)
            pos_embeddings = self.embed(pos_input_ids, pos_attention_mask)
            neg_embeddings = self.embed(neg_input_ids, neg_attention_mask)

            # loss_fn = nn.TripletMarginWithDistanceLoss(
            #     distance_function=self.dist_fn,
            #     margin=self.margin
            # )
            loss = self.compute_loss(sent_embeddings, pos_embeddings, neg_embeddings)
            return loss 
        
        else: # 使用伪数据并且使用我们的做法
            # TODO: 实现我们的做法
            # 第一次前向传播
            sz = sent_input_ids.size(0) // 2
            # 不带梯度情况下计算pos和neg
            with torch.no_grad():
                pos_embeddings = self.embed(pos_input_ids[:sz], pos_attention_mask[:sz])
                neg_embeddings = self.embed(neg_input_ids[:sz], neg_attention_mask[:sz])

            # 我们的模型，先反向传播一次，再根据梯度得到第二波结果
            def backward_hook(module, gin, gout):
                # print('backward function is called.')
                self.emb_grad['grad'] = gout[0].clone().detach() # [batch_size, length, word_dim


            hook = self.roberta.embeddings.word_embeddings.register_full_backward_hook(backward_hook)
            # 前向传播
            sent_embeddings = self.embed(sent_input_ids[:sz], sent_attention_mask[:sz])
            loss = self.compute_loss(sent_embeddings, pos_embeddings, neg_embeddings)

            loss.backward()  # 反向传播
            hook.remove()

            # 计算每个句子对应的token的重要性
            scores = self.emb_grad['grad'].norm(dim=2) # [batch_size, length]
            # print(scores)
            self.emb_grad = {} # 删除grad
            scores = scores / scores.sum(dim=1, keepdims=True) # 所有数值都是正数，归一化到【0，1】

            sent_lens = sent_attention_mask.sum(dim=1)
            if self.config.no_task_desc: # 没有task_desc
                subj_lens = se - ss + 1
                obj_lens = oe - os + 1
            else: # 有task_desc
                subj_lens = se - ss + 1 + desc_se - desc_ss + 1
                obj_lens = oe - os + 1 + desc_oe - desc_os + 1
            ent_lens = subj_lens + obj_lens
            baseline = ent_lens / sent_lens
            sent_scores = []
            for i in range(sz):
                if self.config.no_task_desc:
                    ent_ss, ent_se = ss[i], se[i]
                    ent_os, ent_oe = os[i], oe[i]
                    subj_score = scores[i][ent_ss:ent_se+1].sum()
                    obj_score = scores[i][ent_os:ent_oe+1].sum()
                else:
                    subj_score = scores[i][ss[i]:se[i]+1].sum() + scores[i][desc_ss[i]:desc_se[i]+1].sum()
                    obj_score = scores[i][os[i]:oe[i]+1].sum() + scores[i][desc_os[i]:desc_oe[i]+1].sum()
                sent_score = subj_score + obj_score
                sent_scores.append(sent_score)
            sent_scores = torch.tensor(sent_scores).to(baseline.device)
            # print(sent_scores.shape)
            # print(baseline.shape)
            sent_scores = sent_scores - baseline[:sent_scores.numel()]
            sent_scores = torch.clip(sent_scores, min=0).clone().detach()

            self.zero_grad()

            # with autocast():
            sent_embeddings = self.embed(sent_input_ids, sent_attention_mask)
            pos_embeddings = self.embed(pos_input_ids, pos_attention_mask)
            neg_embeddings = self.embed(neg_input_ids, neg_attention_mask)

            sent_emb1, sent_emb2 = sent_embeddings.chunk(2)
            pos_emb1, pos_emb2 = pos_embeddings.chunk(2)
            neg_emb1, neg_emb2 = neg_embeddings.chunk(2)

            loss1 = self.compute_loss(sent_emb1, pos_emb1, neg_emb1)
            loss2 = self.compute_loss(sent_emb2, pos_emb2, neg_emb2, reduction='none')

            sent_scores = torch.where(torch.isnan(sent_scores), torch.zeros_like(sent_scores), sent_scores)
            loss2 = torch.dot(loss2, sent_scores) / sz
            # 第二次backward
            # loss2.backward()
            # hook.remove() # 结束后记得删除hook
            # print(loss1, loss2)

            # 得到损失返回
            # return loss1 + loss2
            loss = loss1 + loss2
            return loss 
        
        # return loss, sent_embeddings
        
    def compute_importance_score(self, sent_input_ids, pos_input_ids, neg_input_ids, ss, se, os, oe, desc_ss, desc_se, desc_os, desc_oe):
        pass
    
    def embed(
        self,
        input_ids,      
        attention_mask=None,
    ):        
        outputs = self.roberta(input_ids, attention_mask=attention_mask)
        embeddings = outputs[1]              
        
        return embeddings
        
    def dist_fn(
        self,
        sent_embeddings,
        label_embeddings
    ):
        return 1.0 - F.cosine_similarity(sent_embeddings, label_embeddings)