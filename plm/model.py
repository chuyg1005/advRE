import constants
import torch
import torch.nn as nn
import torch.nn.functional as F
from constants import load_constants
from torch.cuda.amp import autocast
from transformers import AutoModel


class REModel(nn.Module):
    def __init__(self, model_name_or_path, num_class, dropout_prob):
        super().__init__()
        # self.args = args
        self.encoder = AutoModel.from_pretrained(model_name_or_path)
        self.encoder.gradient_checkpointing_enable()  # 启用梯度检查点
        hidden_size = self.encoder.config.hidden_size
        self.emb_grad = {}
        # entity_type_rela2id = load_constants(args.data_dir)[2]
        # self.entity_type_rela2id = {}
        # for k, v in entity_type_rela2id.items():
        #     self.entity_type_rela2id[int(k)] = v
        # if self.args.input_format != 'typed_entity_marker_punct_suffix':
        #     self.classifier = nn.Sequential(
        #         nn.Linear(2 * hidden_size, hidden_size),
        #         nn.ReLU(),
        # nn.GELU(),
        # nn.Dropout(p=args.dropout_prob),
        # nn.Linear(hidden_size, args.num_class)
        # )
        self.classifier = nn.Sequential(
            nn.Linear(2 * hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(p=dropout_prob),
            nn.Linear(hidden_size, num_class)
        )

    # def set_encoder_grad(self, requires=False):
    #     for param in self.encoder.parameters():
    #         param.requires_grad = requires

    def save(self, save_path):
        # self.hook.remove()
        torch.save(self, save_path)
        # 重新生成hook
        # def backward_hook(module, gin, gout):
            # print('backward function is called.')
            # self.emb_grad['grad'] = gout[0].clone().detach() # [batch_size, length, word_dim
        # self.hook = self.encoder.embeddings.word_embeddings.register_full_backward_hook(backward_hook)


    # @autocast()  # 自动混合精度训练，提高效率；取消自动混合精度训练
    def forward(self, input_ids=None, attention_mask=None, ss=None, os=None):
        outputs = self.encoder(
            input_ids,
            attention_mask=attention_mask,
        )
        pooled_output = outputs[0]
        idx = torch.arange(input_ids.size(0)).to(input_ids.device)
        ss_emb = pooled_output[idx, ss]
        os_emb = pooled_output[idx, os]
        h = torch.cat((ss_emb, os_emb), dim=-1)
        # if self.args.input_format != 'typed_entity_marker_punct_suffix':
        #     clsemb = pooled_output[:, 0]  # 取出cls向量
        #     ss_emb = pooled_output[idx, ss]
        #     os_emb = pooled_output[idx, os]
        #     h = torch.cat((ss_emb, os_emb), dim=-1)
        # else:
        #     h = pooled_output[idx, mask_idx]
        logits = self.classifier(h)
        logits = logits.float()  # 切换为fp32类型
        return logits
        # mask unknow relas
        # if self.args.mask_rela:
        #     logits = self.mask_logits(logits, subj_type, obj_type)
        # if labels is not None:  # train
            # loss = (loss1 + loss2 * (1 - self.subs_rate)) / (1 + 1 - self.subs_rate)  # 替换比率越高，第二项权重越低
            # loss = self.loss_fnt(logits, labels)
            # 加上kl散度损失
            # probs = F.softmax(logits)
            # prob1, prob2 = probs.chunk(2)
            # prob1, prob2 = F.softmax(logits1), F.softmax(logits2)
            # kl_div = (F.kl_div(prob1, prob2) + F.kl_div(prob2, prob1)) * 0.5
            # loss += self.kl_weight * kl_div  # kl散度, 默认为0.7，0相当于没有kl散度
            # outputs = (loss,) + outputs
            # return outputs
        # else:  # test, mask掉不可能的logits，这个需要决定一下在什么时候使用这个（训练+测试，或者还是只在测试时使用，只在训练时使用不太实际）
        #    logits = self.mask_logits(logits, subj_type, obj_type)
            # return (logits,)

    def compute_bias_score(self, input_ids, attention_mask, ss, se, os,oe, labels):
        batch_sz = input_ids.size(0)
        # 我们的模型，先反向传播一次，再根据梯度得到第二波结果
        def backward_hook(module, gin, gout):
            # print('backward function is called.')
            self.emb_grad['grad'] = gout[0].clone().detach() # [batch_size, length, word_dim
        
        hook = self.encoder.embeddings.word_embeddings.register_full_backward_hook(backward_hook)
        # 前半部分是原始数据，后半部分是增强后的数据
        logits = self(input_ids, attention_mask, ss, os)
        # logits1, logits2 = logits.chunk(2) # 使用kl散度做约束
        # labels1, labels2 = labels.chunk(2) # 使用kl散度约束生成的关系向量
        # 计算真实数据的损失反向传播得到输入的重要性
        loss = F.cross_entropy(logits, labels) 

        # loss1 = self.loss_fnt(logits1, labels1)
        # 添加hook

        # 损失反向传播
        loss.backward()
        hook.remove()
        # 计算每个句子对应的token的重要性
        scores = self.emb_grad['grad'].norm(dim=2) # [batch_size, length]
        # print(scores)
        self.emb_grad = {} # 删除grad
        scores = scores / scores.sum(dim=1, keepdims=True) # 所有数值都是正数，归一化到【0，1】

        sent_lens = attention_mask.sum(dim=1)
        subj_lens = se - ss + 1
        obj_lens = oe - os + 1
        ent_lens = subj_lens + obj_lens
        baseline = ent_lens / sent_lens
        sent_scores = []
        for i in range(batch_sz):
            ent_ss, ent_se = ss[i], se[i]
            ent_os, ent_oe = os[i], oe[i]
            subj_score = scores[i][ent_ss:ent_se+1].sum()
            obj_score = scores[i][ent_os:ent_oe+1].sum()
            sent_score = subj_score + obj_score
            sent_scores.append(sent_score)
        sent_scores = torch.tensor(sent_scores).to(baseline.device)
        # print(sent_scores.shape)
        # print(baseline.shape)
        sent_scores = sent_scores - baseline
        sent_scores = torch.clip(sent_scores, min=0).clone().detach()
        sent_scores = torch.where(torch.isnan(sent_scores), torch.zeros_like(sent_scores), sent_scores)
        self.zero_grad() # 清空梯度

        return sent_scores

    # @torch.no_grad()
    def compute_ours_new_loss(self, input_ids, attention_mask, ss, os, labels):
        sz = input_ids.size(0) // 2
        logits = self(input_ids, attention_mask, ss, os) # [batch_size, label_num]

        # 将 labels 转换为大小为 (batch_size, 1) 的张量
        # labels_new = labels.unsqueeze(1)

        # 使用 torch.gather 函数选择对应标签的 logit
        # selected_logits = torch.gather(logits, 1, labels_new).squeeze(1) # [batch_sz]
        # print(logits.shape)

        # logits1, logits2 = selected_logits.chunk(2)
        # aug = logits1 - logits2 # 越大说明entity-bias越严重
        # org = torch.zeros_like(aug)
        # * 阈值会引入超参数，超参数调节麻烦，因此我们使用统计量减少调节超参数
        # * 让一半的负样本起作用
        # * 如果阈值直接是0结果太差（新实体，旧实体旧组合新关系都比不过，就mask能够比得过）
        # org = torch.full_like(aug, aug.median().item())
        # print(f'median: {aug.median().item()}, mean: {aug.mean().item()}, min: {aug.min().item()}, max: {aug.max().item()}.')
        # weights = torch.stack([org, aug], 0)
        # weights = F.softmax(weights, 0).flatten()

        # weights = weights.clone().detach()

        loss = F.cross_entropy(logits, labels, reduction='none')

        loss1, loss2 = loss.chunk(2)
        aug = loss2 - loss1
        # org = torch.full_like(aug, aug.median().item())
        # org = torch.full_like(aug, aug.quantile(.75).item())
        org = torch.full_like(aug, aug.max().item()) # 切换为max后，伪数据的权重就一定比原始数据的权重低了
        weights = torch.stack([org, aug], 0)
        weights = F.softmax(weights, 0).flatten()
        weights = weights.clone().detach()

        return torch.dot(weights, loss) / sz


    @autocast()
    def compute_loss(self, input_ids=None, attention_mask=None, labels=None, ss=None, se=None, os=None, oe=None, train_mode='baseline'):
        # 基础模型，直接返回
        sz = input_ids.size(0) // 2
        # if use_baseline: # 基础模型
        #     logits = self(input_ids[:sz], attention_mask[:sz], ss[:sz], os[:sz])
        #     loss = F.cross_entropy(logits, labels[:sz]) 
        #     return loss

        # * use_baseline的时候需要传入原始的数据
        if train_mode == 'baseline':
            logits = self(input_ids[:sz], attention_mask[:sz], ss[:sz], os[:sz])
            loss = F.cross_entropy(logits, labels[:sz])
            return loss
        elif train_mode == 'data-aug':
            logits = self(input_ids, attention_mask, ss, os)
            loss = F.cross_entropy(logits, labels)
            return loss
        elif train_mode == 'ours':
            bias_scores = self.compute_bias_score(input_ids[:sz], attention_mask[:sz], ss[:sz], se[:sz], os[:sz], oe[:sz], labels[:sz])
            logits = self(input_ids, attention_mask, ss, os)
            logits1, logits2 = logits.chunk(2)
            labels1, labels2 = labels.chunk(2)
            loss1 = F.cross_entropy(logits1, labels1)
            loss2 = F.cross_entropy(logits2, labels2, reduction='none')

            loss2 = torch.dot(loss2, bias_scores) / sz
            return loss1 + loss2
        elif train_mode == 'ours_new':
            # scores = self.compute_scores(input_ids, attention_mask, ss, os, labels)
            # logits = self(input_ids, attention_mask, ss, os)
            # loss = F.cross_entropy(logits, labels, reduction='none')

            # loss = torch.dot(loss, scores) / sz
            return self.compute_ours_new_loss(input_ids, attention_mask, ss, os, labels)

        else:
            assert 0, f'not support train_mode: {train_mode}.'
            return loss


    # def mask_logits(self, logits, subj_type, obj_type):
    #     mask = torch.full_like(logits, 1e12)
    #     batch_size = logits.size(0)
    #     subj_type = subj_type.cpu().tolist()
    #     obj_type = obj_type.cpu().tolist()
    #     for i in range(batch_size):
    #         key = subj_type[i] * 100 + obj_type[i]
    #         rela_ids = self.entity_type_rela2id[str(key)]
    #         mask[i, rela_ids] = 0.
    #     logits = logits - mask

    #     return logits
    #     return logits
