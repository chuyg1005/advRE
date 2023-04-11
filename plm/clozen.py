"""使用clozen task对实体名称进行mask然后重构"""
import json
import os
from argparse import ArgumentParser

import torch
from tqdm import tqdm
from transformers import AutoModelForMaskedLM, AutoTokenizer
from utils import convert_token


def get_char_pos(tokens, start, end):
    return len(' '.join(tokens[:start])) + 1 if start > 0 else 0, len(' '.join(tokens[:end + 1])) - 1


def process_one_sentence(model, tokenizer, tokens, ss, se, os, oe, device):
    tokens = list(map(convert_token, tokens))
    sentence = ' '.join(tokens)
    char_ss, char_se = get_char_pos(tokens, ss, se)
    char_os, char_oe = get_char_pos(tokens, os, oe)

    # subj_name = sentence[char_ss:char_se + 1]
    # obj_name = sentence[char_os:char_oe + 1]

    encoding = tokenizer(sentence, return_tensors='pt')

    token_ss, token_se = encoding.char_to_token(char_ss), encoding.char_to_token(char_se)
    token_os, token_oe = encoding.char_to_token(char_os), encoding.char_to_token(char_oe)

    input_ids = encoding['input_ids']
    input_ids[0, token_ss:token_se + 1] = tokenizer.mask_token_id
    input_ids[0, token_os:token_oe + 1] = tokenizer.mask_token_id

    out = model(input_ids.to(device))
    out['logits'][0, :, tokenizer.all_special_ids] = -1e12
    out = out['logits'][0].argmax(dim=1).cpu().tolist()
    in_token_ids = input_ids[0].cpu().tolist()

    # out_token_ids = in_token_ids.copy()
    # out_token_ids[token_ss:token_se + 1] = out[token_ss:token_se + 1]
    # out_token_ids[token_os:token_oe + 1] = out[token_os:token_oe + 1]

    # assert len(in_token_ids) == len(out_token_ids) and in_token_ids is not out_token_ids  # 不改变token-ids的长度

    new_subj_token_ids = out[token_ss:token_se + 1]
    new_obj_token_ids = out[token_os:token_oe + 1]

    new_subj_span = tokenizer.decode(new_subj_token_ids).split()
    new_obj_span = tokenizer.decode(new_obj_token_ids).split()

    if ss < os: # subj 出现在ob之前
        new_tokens = tokens[:ss]
        new_ss = len(new_tokens)
        new_tokens += new_subj_span
        new_se = len(new_tokens) - 1
        new_tokens += tokens[se+1:os]
        new_os = len(new_tokens)
        new_tokens += new_obj_span
        new_oe = len(new_tokens) - 1
        new_tokens += tokens[oe+1:]
    else:
        new_tokens = tokens[:os]
        new_os = len(new_tokens)
        new_tokens += new_obj_span
        new_oe = len(new_tokens) - 1
        new_tokens += tokens[oe+1:ss]
        new_ss = len(new_tokens)
        new_tokens += new_subj_span
        new_se = len(new_tokens) - 1
        new_tokens += tokens[se+1:]

    return new_tokens, new_ss, new_se, new_os, new_oe
    #
    # new_sentence = tokenizer.decode(out_token_ids, skip_special_tokens=True,
    #                                 clean_up_tokenization_spaces=False)  # 获取新句子
    # new_sentence = tokenizer.decode(out_token_ids, skip_special_tokens=True)

    # new_encoding = tokenizer(new_sentence)

    # if new_encoding['input_ids'] != out_token_ids:
    #     print('miss error.')
    #     return tokens, ss, se, os, oe
    # assert new_encoding['input_ids'] == out_token_ids  # 解码再编码结果一致

    # new_char_ss, new_char_se = new_encoding.token_to_chars(token_ss)[0], new_encoding.token_to_chars(token_se)[1] - 1
    # new_char_os, new_char_oe = new_encoding.token_to_chars(token_os)[0], new_encoding.token_to_chars(token_oe)[1] - 1

    # new_char_ss, new_char_os = new_sentence.index(subj_name), new_sentence.index(obj_name)
    # new_char_se, new_char_oe = new_char_ss + len(subj_name) - 1, new_char_os + len(obj_name) - 1

    # new_tokens = new_sentence.split()
    # new_ss, new_os = len(new_sentence[:new_char_ss].split()), len(new_sentence[:new_char_os].split())
    # new_se, new_oe = max(len(new_sentence[:new_char_se].split()) - 1, 0), max(
    #     len(new_sentence[:new_char_oe].split()) - 1, 0)
    #
    # if new_ss > new_se: new_ss = new_se
    # if new_os > new_oe: new_os = new_oe
    #
    # return new_tokens, new_ss, new_se, new_os, new_oe
    #
    # new_encoding = tokenizer(new_sentence)
    # assert new_encoding['input_ids'] == out_token_ids  # 解码再编码结果一致
    #
    # new_char_ss, new_char_se = encoding.token_to_char(token_ss), encoding.token_to_char(token_se)
    # new_char_os, new_char_oe = encoding.token_to_char(token_os), encoding.token_to_char(token_oe)


def main(args):
    # model_name = 'bert-base-cased'
    # data_path = 'data/tacred/train.json'
    # data_dir = 'data/tacred'
    model_name = args.model_name
    data_dir = args.data_dir

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    out_dir = os.path.join(data_dir, model_name)
    os.makedirs(out_dir, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained(model_name).to(device)
    model.eval()

    for filename in ['train.json', 'dev.json', 'test.json']:
        data = json.load(open(os.path.join(data_dir, filename)))
        new_data = []
        change_cnt = 0
        for item in tqdm(data):
            new_tokens, new_ss, new_se, new_os, new_oe = \
                process_one_sentence(model, tokenizer, item['token'],
                                     item['subj_start'], item['subj_end'],
                                     item['obj_start'], item['obj_end'], device)

            if item['token'] != new_tokens: change_cnt += 1
            item['token'] = new_tokens
            item['subj_start'], item['subj_end'] = new_ss, new_se
            item['obj_start'], item['obj_end'] = new_os, new_oe

            new_data.append(item)

        print(f'{filename} has been processed, [{change_cnt} / {len(data)}] changes.')
        with open(os.path.join(out_dir, filename), 'w') as f:
            json.dump(new_data, f)


if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--data_dir', type=str, required=True)

    args = parser.parse_args()

    main(args)
