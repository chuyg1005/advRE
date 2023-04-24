import os

import ujson as json
# import constants
from constants import load_constants
from text_encoder import TextEncoder
from tqdm import tqdm
from transformers import AutoTokenizer
from utils import convert_token


class Processor:
    def __init__(self, input_format, tokenizer, max_seq_length, rela2id):
        super().__init__()
        # self.args = args
        self.input_format = input_format
        self.max_seq_length = max_seq_length
        self.tokenizer = tokenizer
        self.LABEL_TO_ID = rela2id
        # self.entity_type2id = entity_type2id
        # self.use_pseudo = use_pseudo
        self.text_encoder = TextEncoder.build_text_encoder(input_format, tokenizer, max_seq_length)

    def tokenize(self, tokens, subj_type, obj_type, ss, se, os, oe, mask_rate, all):
        """
        Implement the following input formats:
            - entity_mask: [SUBJ-NER], [OBJ-NER].
            - entity_marker: [E1] subject [/E1], [E2] object [/E2].
            - entity_marker_punct: @ subject @, # object #.
            - typed_entity_marker: [SUBJ-NER] subject [/SUBJ-NER], [OBJ-NER] obj [/OBJ-NER]
            - typed_entity_marker_punct: @ * subject ner type * subject @, # ^ object ner type ^ object #
        """
        return self.text_encoder.mask_encode(tokens, subj_type, obj_type, ss, se, os, oe, mask_rate, all)
        # if not mask:
        #     return self.text_encoder.encode(tokens, subj_type, obj_type, ss, se, os, oe)
        # else:
        #     return self.text_encoder.mask_encode(tokens, subj_type, obj_type, ss, se, os, oe)
    def get_feature(self, d, mask_rate=0., all=False):
        # mask表示是否需要mask住实体名称
        ss, se = d['subj_start'], d['subj_end']
        os, oe = d['obj_start'], d['obj_end']

        tokens = d['token']
        tokens = [convert_token(token) for token in tokens]

        input_ids, new_ss, new_se, new_os, new_oe = self.tokenize(tokens, d['subj_type'], d['obj_type'], ss, se, os,
                                                                    oe, mask_rate, all)
        # subj_type, obj_type = self.entity_type2id[d['subj_type']], self.entity_type2id[d['obj_type']]
        rel = self.LABEL_TO_ID[d['relation']]

        # subj, obj = input_ids[new_ss:new_se + 1], input_ids[new_os:new_oe + 1]
        # if subj_type not in subj_dict: subj_dict[subj_type] = set()
        # if obj_type not in obj_dict: obj_dict[obj_type] = set()

        # subj_dict[subj_type].add(tuple(subj))
        # obj_dict[obj_type].add(tuple(obj))

        feature = {
            'input_ids': input_ids,
            'labels': rel,
            'ss': new_ss,  # subj开始的编号
            'se': new_se,  # subj结束的编号
            'os': new_os,  # obj开始的编号
            'oe': new_oe,  # obj结束的编号
            # 'subj_type': subj_type,
            # 'obj_type': obj_type,
            # 'mask_idx': mask_idx
        }
        return feature

    def read(self, file_in, mode='train', mask_rate=0., all=False):
        features = []
        with open(file_in, "r") as fh:
            data = json.load(fh)

        # subj_dict = {}
        # obj_dict = {}

        # 训练模式下带伪数据的
        if mode == 'train': # 训练模式并且每个具有两个样本
            for d in tqdm(data):
                feature = []
                for sample in d:
                    f = self.get_feature(sample, mask_rate, all)
                    feature.append(f)
                features.append(feature)
        else:
            for d in tqdm(data):
                feature = self.get_feature(d, mask_rate, all)
                features.append(feature)
        
        return features

        # if mode == 'train':
        #     new_subj_dict, new_obj_dict = {}, {}
        #     for subj_type in subj_dict:
        #         new_subj_dict[str(subj_type)] = list(subj_dict[subj_type])
        #     for obj_type in obj_dict:
        #         new_obj_dict[str(obj_type)] = list(obj_dict[obj_type])
        #     return features, new_subj_dict, new_obj_dict
        # else:
        #     return features


if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
    input_format = 'typed_entity_marker_punct'
    max_seq_length = 128
    data_dir = './data/tacred'

    train_file = os.path.join(data_dir, "train.json")
    processor = Processor(input_format, max_seq_length, data_dir, tokenizer)
    train_features = processor.read(train_file)
