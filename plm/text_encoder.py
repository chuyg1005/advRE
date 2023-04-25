import json
import random

import numpy as np
from transformers import AutoTokenizer
from utils import convert_token


class TextEncoder:
    @staticmethod
    def build_text_encoder(input_format, tokenizer, max_seq_length, subj_types=None, obj_types=None):
        # subj_types / obj_types用来增加新的token
        if input_format == 'entity_name_only':  # 只有实体名称
            return EntityNameOnlyTextEncoder(input_format, tokenizer, max_seq_length)
        elif input_format == 'entity_type_mask':
            return EntityMaskTextEncoder(input_format, tokenizer, max_seq_length, subj_types, obj_types)
        elif input_format == 'entity_mask_new': # 只有上下文信息
            return EntityMaskTextEncoderNew(input_format, tokenizer, max_seq_length, subj_types, obj_types)
        elif input_format == 'entity_marker':
            return EntityMarkerTextEncoder(input_format, tokenizer, max_seq_length)
        elif input_format == 'entity_marker_new':
            return EntityMarkerTextEncoderNew(input_format, tokenizer, max_seq_length)
        elif input_format == 'entity_marker_punct_new': # 第一个（只有实体的名称）
            return EntityMarkerTextEncoderNew(input_format, tokenizer, max_seq_length, True)
        elif input_format == 'typed_entity_marker':
            return TypedEntityMarkerTextEncoder(input_format, tokenizer, max_seq_length, subj_types, obj_types)
        elif input_format == 'typed_entity_marker_new':
            return TypedEntityMarkerTextEncoderNew(input_format, tokenizer, max_seq_length, subj_types, obj_types)
        elif input_format == 'typed_entity_marker_punct':
            return TypedEntityMarkerPunctTextEncoder(input_format, tokenizer, max_seq_length)
        elif input_format == 'typed_entity_marker_punct_new': # 第二个（同时有实体的名称和实体的类型）
            return TypedEntityMarkerPunctNewTextEncoder(input_format, tokenizer, max_seq_length)
        # elif input_format == 'typed_entity_marker_punct_mask_new':  # 提供实体的类型信息但是mask住了实体名称，用于测试新实体下的性能
        #     return TypedEntityMarkerPunctNewTextEncoder(input_format, tokenizer, max_seq_length, True)
        # elif input_format == 'typed_entity_marker_punct_mask_new':
        #     return TypedEntityMarkerPunctNewTextEncoder(input_format, tokenizer, max_seq_length, mask=True)
        elif input_format == 'entity_type_marker':
            return EntityTypeMarkerTextEncoder(input_format, tokenizer, max_seq_length)
        elif input_format == 'entity_type_marker_new':
            return EntityTypeMarkerTextEncoderNew(input_format, tokenizer, max_seq_length)
        elif input_format == 'entity_type_marker_punct_new': # 第三个（只有实体的类型信息）
            return EntityTypeMarkerTextEncoderNew(input_format, tokenizer, max_seq_length, True)
        elif input_format == 'entity_type_marker_punct_ext_new':
            return EntityTypeMarkerTextEncoderPunctExtNew(input_format, tokenizer, max_seq_length, True)
        else:
            raise Exception('Invalid input format!')

    def __init__(self, input_format, tokenizer, max_seq_length):
        self.input_format = input_format
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.new_tokens = []
        # self.new_tokens = self.get_new_tokens(subj_types, obj_types)
        # self.tokenizer.add_tokens(self.new_tokens)

    def encode(self, tokens, subj_type, obj_type, ss, se, os, oe):
        pass


    def mask_encode(self, tokens, subj_type, obj_type, ss, se, os, oe, mask_rate=1., all=False):
        # mask住实体名称来编码，用于测试阶段
        return self.encode(tokens, subj_type, obj_type, ss, se, os, oe)

    # def get_new_tokens(self, subj_types, obj_types):
    #     return []


class EntityMaskTextEncoder(TextEncoder):
    def __init__(self, input_format, tokenizer, max_seq_length, subj_types, obj_types):
        super().__init__(input_format, tokenizer, max_seq_length)
        # 添加新的token
        new_tokens = set()
        for subj_type in subj_types:
            new_tokens.add('[SUBJ-{}]'.format(subj_type))
        for obj_type in obj_types:
            new_tokens.add('[OBJ-{}]'.format(obj_type))
        self.new_tokens = list(new_tokens)
        self.tokenizer.add_tokens(self.new_tokens)
    def encode(self, tokens, subj_type, obj_type, ss, se, os, oe):
        sents = []
        subj_type = '[SUBJ-{}]'.format(subj_type)
        obj_type = '[OBJ-{}]'.format(obj_type)
        new_char_ss, new_char_os = -1, -1
        for i_t, token in enumerate(tokens):
            if ss <= i_t <= se or os <= i_t <= oe:
                if i_t == ss:
                    sents.append(subj_type)
                    new_char_ss = len(' '.join(sents)) - 1
                if i_t == os:
                    sents.append(obj_type)
                    new_char_os = len(' '.join(sents)) - 1
            else:
                sents.append(token)

        encoding = self.tokenizer.encode_plus(' '.join(sents))
        new_ss, new_os = encoding.char_to_token(new_char_ss), encoding.char_to_token(new_char_os)

        input_ids = encoding['input_ids'][:self.max_seq_length]
        # input_ids[new_ss] = self.tokenizer.mask_token_id
        # input_ids[new_os] = self.tokenizer.mask_token_id
        # 直接将实体名称替换为mask，不使用实体类型和实体名称信息

        return input_ids, new_ss, new_ss, new_os, new_os

    # def encode(self, tokens, subj_type, obj_type, ss, se, os, oe):
    #     sents = []
    #     subj_type = '[SUBJ-{}]'.format(subj_type)
    #     obj_type = '[OBJ-{}]'.format(obj_type)

    #     for i_t, token in enumerate(tokens):
    #         tokens_wordpiece = self.tokenizer.tokenize(token)
    #         if ss <= i_t <= se or os <= i_t <= oe:
    #             tokens_wordpiece = []
    #             if i_t == ss:
    #                 new_ss = len(sents)
    #                 tokens_wordpiece = [subj_type]
    #             if i_t == os:
    #                 new_os = len(sents)
    #                 tokens_wordpiece = [obj_type]
    #         sents.extend(tokens_wordpiece)

    #     sents = sents[:self.max_seq_length - 2]
    #     input_ids = self.tokenizer.convert_tokens_to_ids(sents)
    #     input_ids = self.tokenizer.build_inputs_with_special_tokens(input_ids)
    #     # ss, se, os ,oe
    #     return input_ids, new_ss + 1, new_ss + 1, new_os + 1, new_os + 1  # [CLS] offset


class EntityMaskTextEncoderNew(TextEncoder):
    def __init__(self, input_format, tokenizer, max_seq_length, subj_types, obj_types):
        super().__init__(input_format, tokenizer, max_seq_length)
        self.new_tokens = ['[SUBJ]', '[OBJ]']
        self.tokenizer.add_tokens(self.new_tokens)

    # def mask_encode(self, tokens, subj_type, obj_type, ss, se, os, oe, mask_rate.):
    #     return self.encode(tokens, subj_type, obj_type, ss, se, os, oe)

    def encode(self, tokens, subj_type, obj_type, ss, se, os, oe):
        sents = []
        # subj_type = '[SUBJ-{}]'.format(subj_type)
        # obj_type = '[OBJ-{}]'.format(obj_type)
        subj_type = '[SUBJ]'
        obj_type = '[OBJ]'
        new_char_ss, new_char_os = -1, -1
        for i_t, token in enumerate(tokens):
            if ss <= i_t <= se or os <= i_t <= oe:
                if i_t == ss:
                    sents.append(subj_type)
                    new_char_ss = len(' '.join(sents)) - 1
                if i_t == os:
                    sents.append(obj_type)
                    new_char_os = len(' '.join(sents)) - 1
            else:
                sents.append(token)

        encoding = self.tokenizer.encode_plus(' '.join(sents))
        new_ss, new_os = encoding.char_to_token(new_char_ss), encoding.char_to_token(new_char_os)

        input_ids = encoding['input_ids'][:self.max_seq_length]
        # input_ids[new_ss] = self.tokenizer.mask_token_id
        # input_ids[new_os] = self.tokenizer.mask_token_id
        # 直接将实体名称替换为mask，不使用实体类型和实体名称信息

        return input_ids, new_ss, new_ss, new_os, new_os
        # return encoding['input_ids'][:self.max_seq_length], new_ss, new_os
        # return encoding['input_ids'][:self.max_seq_length], new_ss, new_ss, new_os, new_os


class EntityTypeMarkerTextEncoder(TextEncoder):
    def __init__(self, input_format, tokenizer, max_seq_length, use_punct=False):
        super().__init__(input_format, tokenizer, max_seq_length)
        self.use_punct = use_punct
        if use_punct:
            self.ss, self.se, self.os, self.oe = '@', '@', '#', '#'
        else:
            self.ss, self.se, self.os, self.oe = '[E1]', '[/E1]', '[E2]', '[/E2]'
            self.new_tokens = ['[E1]', '[/E1]', '[E2]', '[/E2]']
            self.tokenizer.add_tokens(self.new_tokens)

    def encode(self, tokens, subj_type, obj_type, ss, se, os, oe):
        subj_type = self.tokenizer.tokenize(subj_type.replace("_", " ").lower())
        obj_type = self.tokenizer.tokenize(obj_type.replace("_", " ").lower())
        sents = []
        for i_t, token in enumerate(tokens):
            tokens_wordpiece = self.tokenizer.tokenize(token)
            if ss <= i_t <= se or os <= i_t <= oe:
                tokens_wordpiece = []
                if i_t == ss:
                    new_ss = len(sents)
                    tokens_wordpiece = [self.ss] + subj_type + [self.se]
                    new_se = len(sents) + len(tokens_wordpiece) - 1
                if i_t == os:
                    new_os = len(sents)
                    tokens_wordpiece = [self.os] + obj_type + [self.oe]
                    new_oe = len(sents) + len(tokens_wordpiece) - 1

            sents.extend(tokens_wordpiece)

        sents = sents[:self.max_seq_length - 2]
        input_ids = self.tokenizer.convert_tokens_to_ids(sents)
        input_ids = self.tokenizer.build_inputs_with_special_tokens(input_ids)
        return input_ids, new_ss + 1, new_se + 1, new_os + 1, new_oe + 1  # [CLS] offset


class EntityTypeMarkerTextEncoderNew(EntityTypeMarkerTextEncoder):
    # def mask_encode(self, tokens, subj_type, obj_type, ss, se, os, oe, mask_rate=1.):
    #     # 不带有实体名称信息
    #     return self.encode(tokens, subj_type, obj_type, ss, se, os, oe)
    
    def encode(self, tokens, subj_type, obj_type, ss, se, os, oe):
        subj_type = subj_type.replace("_", " ").lower().split()
        obj_type = obj_type.replace("_", " ").lower().split()
        sents = []
        new_char_ss, new_char_os = -1, -1
        new_char_se, new_char_oe = -1, -1
        for i_t, token in enumerate(tokens):
            if ss <= i_t <= se or os <= i_t <= oe:
                if i_t == ss:
                    new_char_ss = len(' '.join(sents)) + 1 if sents else 1  # 空格
                    if sents:
                        sents += [self.ss] + subj_type + [self.se]
                    else:
                        sents = [' ' + self.ss] + subj_type + [self.se]
                    new_char_se = len(' '.join(sents)) - 1  # 最后一个
                if i_t == os:
                    new_char_os = len(' '.join(sents)) + 1 if sents else 1  # 空格
                    if sents:
                        sents += [self.os] + obj_type + [self.oe]
                    else:
                        sents = [' ' + self.os] + obj_type + [self.oe]
                    new_char_oe = len(' '.join(sents)) - 1  # 最后一个
            else:
                sents.append(token)
        encoding = self.tokenizer.encode_plus(' '.join(sents))
        new_ss, new_os = encoding.char_to_token(new_char_ss), encoding.char_to_token(new_char_os)
        new_se, new_oe = encoding.char_to_token(new_char_se), encoding.char_to_token(new_char_oe)

        # return encoding['input_ids'][:self.max_seq_length], new_ss, new_os
        return encoding['input_ids'][:self.max_seq_length], new_ss, new_se, new_os, new_oe


class EntityTypeMarkerTextEncoderPunctExtNew(EntityTypeMarkerTextEncoder):
    """加上类型标签，同时保留实体名称，不在实体名称边界加内容"""

    def encode(self, tokens, subj_type, obj_type, ss, se, os, oe):
        subj_type = subj_type.replace("_", " ").lower().split()
        obj_type = obj_type.replace("_", " ").lower().split()
        sents = []
        new_char_ss, new_char_os = -1, -1
        new_char_se, new_char_oe = -1, -1
        for i_t, token in enumerate(tokens):
            if i_t == ss:
                new_char_ss = len(' '.join(sents)) + 1 if sents else 0  # 空格
                sents += [self.ss] + subj_type + [self.se]
                new_char_se = len(' '.join(sents)) - 1  # 最后一个
            if i_t == os:
                new_char_os = len(' '.join(sents)) + 1 if sents else 0  # 空格
                sents += [self.os] + obj_type + [self.oe]
                new_char_oe = len(' '.join(sents)) - 1  # 最后一个
            sents.append(token)
        encoding = self.tokenizer.encode_plus(' '.join(sents))
        new_ss, new_os = encoding.char_to_token(new_char_ss), encoding.char_to_token(new_char_os)
        new_se, new_oe = encoding.char_to_token(new_char_se), encoding.char_to_token(new_char_oe)

        assert new_ss, 'subject start position is None.'
        assert new_se, 'subject end position is None.'
        assert new_os, 'object start position is None.'
        assert new_oe, 'object end position is None.'

        # return encoding['input_ids'][:self.max_seq_length], new_ss, new_os
        return encoding['input_ids'][:self.max_seq_length], new_ss, new_se, new_os, new_oe


class EntityMarkerTextEncoder(TextEncoder):
    def __init__(self, input_format, tokenizer, max_seq_length, use_punct=False):
        super().__init__(input_format, tokenizer, max_seq_length)
        if use_punct:
            self.ss, self.se, self.os, self.oe = '@', '@', '#', '#'
        else:
            self.ss, self.se, self.os, self.oe = '[E1]', '[/E1]', '[E2]', '[/E2]'
            self.new_tokens = ['[E1]', '[/E1]', '[E2]', '[/E2]']
            self.tokenizer.add_tokens(self.new_tokens)  # 重复添加token会被过滤掉

    def encode(self, tokens, subj_type, obj_type, ss, se, os, oe):
        sents = []
        for i_t, token in enumerate(tokens):
            tokens_wordpiece = self.tokenizer.tokenize(token)
            if i_t == ss:
                new_ss = len(sents)
                tokens_wordpiece = [self.ss] + tokens_wordpiece
            if i_t == se:
                tokens_wordpiece = tokens_wordpiece + [self.se]
                new_se = len(sents) + len(tokens_wordpiece) - 1
            if i_t == os:
                new_os = len(sents)
                tokens_wordpiece = [self.os] + tokens_wordpiece
            if i_t == oe:
                tokens_wordpiece = tokens_wordpiece + [self.oe]
                new_oe = len(sents) + len(tokens_wordpiece) - 1

            sents.extend(tokens_wordpiece)

        sents = sents[:self.max_seq_length - 2]
        input_ids = self.tokenizer.convert_tokens_to_ids(sents)
        input_ids = self.tokenizer.build_inputs_with_special_tokens(input_ids)
        return input_ids, new_ss + 1, new_se + 1, new_os + 1, new_oe + 1  # [CLS] offset



class EntityMarkerTextEncoderNew(EntityMarkerTextEncoder):
    def encode(self, tokens, subj_type, obj_type, ss, se, os, oe):
        sents = []
        new_char_ss, new_char_os = -1, -1
        new_char_se, new_char_oe = -1, -1
        for i_t, token in enumerate(tokens):
            if i_t == ss:
                sents.append(self.ss)
                new_char_ss = len(' '.join(sents)) - 1
                sents.append(token)
            if i_t == se:
                sents.append(token)
                sents.append(self.se)
                new_char_se = len(' '.join(sents)) - 1
            if i_t == os:
                sents.append(self.os)
                new_char_os = len(' '.join(sents)) - 1
                sents.append(token)
            if i_t == oe:
                sents.append(token)
                sents.append(self.oe)
                new_char_oe = len(' '.join(sents)) - 1
            if i_t != ss and i_t != se and i_t != os and i_t != oe:
                sents.append(token)

        encoding = self.tokenizer.encode_plus(' '.join(sents))
        new_ss, new_os = encoding.char_to_token(new_char_ss), encoding.char_to_token(new_char_os)
        new_se, new_oe = encoding.char_to_token(new_char_se), encoding.char_to_token(new_char_oe)

        # return encoding['input_ids'][:self.max_seq_length], new_ss, new_os
        return encoding['input_ids'][:self.max_seq_length], new_ss, new_se, new_os, new_oe

    def mask_encode(self, tokens, subj_type, obj_type, ss, se, os, oe, mask_rate=1., all=False):
        mask_token = self.tokenizer.mask_token # 获取mask_token
        # 进行token的替换
        subj_len = se - ss + 1
        obj_len = oe - os + 1

        if not all:
            subj_masks = np.random.choice(subj_len, int(subj_len * mask_rate), replace=False)
            obj_masks = np.random.choice(obj_len, int(obj_len * mask_rate), replace=False)

            for idx in subj_masks:
                tokens[ss+idx] = mask_token

            for idx in obj_masks:
                tokens[os+idx] = mask_token
        else:
            n = subj_len + obj_len
            mask_num = int(n * mask_rate)
            masks = np.random.choice(len(tokens), mask_num, replace=False)

            for idx in masks:
                tokens[idx] = mask_token
        # print(tokens)
        # tokens[ss:se+1] = [mask_token] * subj_len
        # tokens[os:oe+1] = [mask_token] * obj_len

        return self.encode(tokens, subj_type, obj_type, ss, se, os, oe)

    # def mask_encode(self, tokens, subj_type, obj_type, ss, se, os, oe, mask_rate=1., all=False):
    #     # mask住实体模拟实体信息丢失的情况
        # input_ids, ss, se, os, oe = self.encode(tokens, subj_type, obj_type, ss, se, os, oe)

        # subj_len = se - ss - 1 # subj长度
        # obj_len = oe - os - 1 # obj长度
        # if not all:
        #     subj_masks = np.random.choice(subj_len, int(subj_len * mask_rate), replace=False)
        #     obj_masks = np.random.choice(obj_len, int(obj_len * mask_rate), replace=False)

        #     for idx in subj_masks:
        #         input_ids[ss+1+idx] = self.tokenizer.mask_token_id
        #     for idx in obj_masks:
        #         input_ids[os+1+idx] = self.tokenizer.mask_token_id
        # else:
        #     n = subj_len + obj_len
        #     mask_num = int(n * mask_rate) # mask
        #     masks = np.random.choice(len(input_ids), mask_num, replace=False)

        #     for idx in masks:
        #         input_ids[idx] = self.tokenizer.mask_token_id

        # # input_ids[ss+1:se] = [self.tokenizer.mask_token_id] * subj_len
        # # input_ids[os+1:oe] = [self.tokenizer.mask_token_id] * obj_len

        # return input_ids, ss, se, os, oe

class EntityNameOnlyTextEncoder(EntityMarkerTextEncoderNew):
    def __init__(self, input_format, tokenizer, max_seq_length):
        super().__init__(input_format, tokenizer, max_seq_length, True)
        # new_tokens = ['[SUBJ]', '[OBJ]']
        # self.new_tokens += new_tokens
        # self.tokenizer.add_tokens(new_tokens)

    def mask_encode(self, tokens, subj_type, obj_type, ss, se, os, oe, mask_rate=1, all=False):
        # if mask_rate == 0:
            # return self.encode(tokens, subj_type, obj_type, ss, se, os, oe)
        # assert mask_rate == 1
        # mask_token = self.tokenizer.mask_token
        # tokens = [mask_token, 'and', mask_token]
        # ss, se = 0, 0
        # os, oe = 2, 2
        return self.encode(tokens, subj_type, obj_type, ss, se ,os, oe)

    def encode(self, tokens, subj_type, obj_type, ss, se, os, oe):
        subj = tokens[ss:se+1]
        obj = tokens[os:oe+1]
        tokens = subj + ['and'] + obj
        ss, se = 0, len(subj) - 1
        os, oe = len(subj) + 1, len(tokens) - 1
        # print('inner method.')
        # print(tokens)

        return super().encode(tokens, subj_type, obj_type, ss, se, os, oe)


class TypedEntityMarkerTextEncoder(TextEncoder):
    def __init__(self, input_format, tokenizer, max_seq_length, subj_types, obj_types):
        super().__init__(input_format, tokenizer, max_seq_length)
        new_tokens = set()
        for subj_type in subj_types:
            new_tokens.add('[SUBJ-{}]'.format(subj_type))
            new_tokens.add('[/SUBJ-{}]'.format(subj_type))
        for obj_type in obj_types:
            new_tokens.add('[OBJ-{}]'.format(obj_type))
            new_tokens.add('[/OBJ-{}]'.format(obj_type))
        self.new_tokens = list(new_tokens)
        self.tokenizer.add_tokens(self.new_tokens)

    def encode(self, tokens, subj_type, obj_type, ss, se, os, oe):
        sents = []
        subj_start = '[SUBJ-{}]'.format(subj_type)
        subj_end = '[/SUBJ-{}]'.format(subj_type)
        obj_start = '[OBJ-{}]'.format(obj_type)
        obj_end = '[/OBJ-{}]'.format(obj_type)
        for i_t, token in enumerate(tokens):
            tokens_wordpiece = self.tokenizer.tokenize(token)
            if i_t == ss:
                new_ss = len(sents)
                tokens_wordpiece = [subj_start] + tokens_wordpiece
            if i_t == se:
                tokens_wordpiece = tokens_wordpiece + [subj_end]
                new_se = len(sents) + len(tokens_wordpiece) - 1
            if i_t == os:
                new_os = len(sents)
                tokens_wordpiece = [obj_start] + tokens_wordpiece
            if i_t == oe:
                tokens_wordpiece = tokens_wordpiece + [obj_end]
                new_oe = len(sents) + len(tokens_wordpiece) - 1

            sents.extend(tokens_wordpiece)

        sents = sents[:self.max_seq_length - 2]
        input_ids = self.tokenizer.convert_tokens_to_ids(sents)
        input_ids = self.tokenizer.build_inputs_with_special_tokens(input_ids)
        return input_ids, new_ss + 1, new_se + 1, new_os + 1, new_oe + 1  # [CLS] offset


class TypedEntityMarkerTextEncoderNew(TypedEntityMarkerTextEncoder):
    def encode(self, tokens, subj_type, obj_type, ss, se, os, oe):
        sents = []
        subj_start = '[SUBJ-{}]'.format(subj_type)
        subj_end = '[/SUBJ-{}]'.format(subj_type)
        obj_start = '[OBJ-{}]'.format(obj_type)
        obj_end = '[/OBJ-{}]'.format(obj_type)
        new_char_ss, new_char_os = -1, -1
        new_char_se, new_char_oe = -1, -1
        for i_t, token in enumerate(tokens):
            if i_t == ss:
                sents.append(subj_start)
                new_char_ss = len(' '.join(sents)) - 1
                sents.append(token)
            if i_t == se:
                sents.append(token)
                sents.append(subj_end)
                new_char_se = len(' '.join(sents)) - 1
            if i_t == os:
                sents.append(obj_start)
                new_char_os = len(' '.join(sents)) - 1
                sents.append(token)
            if i_t == oe:
                sents.append(token)
                sents.append(obj_end)
                new_char_oe = len(' '.join(sents)) - 1
            if i_t != ss and i_t != se and i_t != os and i_t != oe:
                sents.append(token)

        encoding = self.tokenizer.encode_plus(' '.join(sents))
        new_ss, new_os = encoding.char_to_token(new_char_ss), encoding.char_to_token(new_char_os)
        new_se, new_oe = encoding.char_to_token(new_char_se), encoding.char_to_token(new_char_oe)

        return encoding['input_ids'][:self.max_seq_length], new_ss, new_se, new_os, new_oe


class TypedEntityMarkerPunctTextEncoder(TextEncoder):
    # def __init__(self):
    #     pass

    def encode(self, tokens, subj_type, obj_type, ss, se, os, oe):
        subj_type = self.tokenizer.tokenize(subj_type.replace("_", " ").lower())
        obj_type = self.tokenizer.tokenize(obj_type.replace("_", " ").lower())
        sents = []
        for i_t, token in enumerate(tokens):
            tokens_wordpiece = self.tokenizer.tokenize(token)
            if i_t == ss:
                new_ss = len(sents)
                tokens_wordpiece = ['@'] + ['*'] + subj_type + ['*'] + tokens_wordpiece
            if i_t == se:
                tokens_wordpiece = tokens_wordpiece + ['@']
                new_se = len(sents) + len(tokens_wordpiece) - 1
            if i_t == os:
                new_os = len(sents)
                tokens_wordpiece = ["#"] + ['^'] + obj_type + ['^'] + tokens_wordpiece
            if i_t == oe:
                tokens_wordpiece = tokens_wordpiece + ["#"]
                new_oe = len(sents) + len(tokens_wordpiece) - 1

            sents += tokens_wordpiece
        sents = sents[:self.max_seq_length - 2]
        input_ids = self.tokenizer.convert_tokens_to_ids(sents)
        input_ids = self.tokenizer.build_inputs_with_special_tokens(input_ids)
        return input_ids, new_ss + 1, new_se + 1, new_os + 1, new_oe + 1


class TypedEntityMarkerPunctNewTextEncoder(TypedEntityMarkerPunctTextEncoder):
    def __init__(self, input_format, tokenizer, max_seq_length):
        super().__init__(input_format, tokenizer, max_seq_length)
        # self.mask = mask

    # def __init__(self, input_format, tokenizer, max_seq_length, mask=False, mask_ratio=0.7):
    # super().__init__(input_format, tokenizer, max_seq_length)
    # self.mask = mask
    # self.mask_ratio = 0.7

    # def mask_entity(self, tokens, ent1_s, ent1_e, ent2_s, ent2_e, mask_token):
    #     assert ent1_e < ent2_s or ent2_e < ent1_s, 'entities overlapped with each other!'
    #     ent1_len = ent1_e - ent1_s + 1
    #     tokens = tokens[:ent1_s] + [mask_token] + tokens[ent1_e + 1:]
    #     ent1_e = ent1_s
    #     if ent1_e < ent2_s:
    #         ent2_s -= ent1_len - 1
    #         ent2_e -= ent1_len - 1
    #     return tokens, ent1_s, ent1_e, ent2_s, ent2_e

    def mask_encode(self, tokens, subj_type, obj_type, ss, se, os, oe, mask_rate=1., all=False):
        mask_token = self.tokenizer.mask_token # 获取mask_token
        # 进行token的替换
        subj_len = se - ss + 1
        obj_len = oe - os + 1

        if not all:
            subj_masks = np.random.choice(subj_len, int(subj_len * mask_rate), replace=False)
            obj_masks = np.random.choice(obj_len, int(obj_len * mask_rate), replace=False)

            for idx in subj_masks:
                tokens[ss+idx] = mask_token

            for idx in obj_masks:
                tokens[os+idx] = mask_token
        else:
            n = subj_len + obj_len
            mask_num = int(n * mask_rate)
            masks = np.random.choice(len(tokens), mask_num, replace=False)

            for idx in masks:
                tokens[idx] = mask_token
        # tokens[ss:se+1] = [mask_token] * subj_len
        # tokens[os:oe+1] = [mask_token] * obj_len

        return self.encode(tokens, subj_type, obj_type, ss, se, os, oe)

    def encode(self, tokens, subj_type, obj_type, ss, se, os, oe):
        sents = []
        new_char_ss, new_char_se = -1, -1
        new_char_os, new_char_oe = -1, -1
        # name_start, name_end
        subj_type = subj_type.replace('_', ' ').lower().split()
        obj_type = obj_type.replace('_', ' ').lower().split()
        # rand1, rand2 = random.random(), random.random()
        # mask_subj = rand1 < self.mask_ratio if self.mask else False
        # mask_obj = rand2 < self.mask_ratio if self.mask else False
        # print(tokens)
        # if self.mask: # mask住实体名称
        #     tokens, ss, se, os, oe = self.mask_entity(tokens, ss, se, os, oe, self.tokenizer.mask_token)
        #     tokens, os, oe, ss, se = self.mask_entity(tokens, os, oe, ss, se, self.tokenizer.mask_token)
        # # print(tokens)
        for i_t, token in enumerate(tokens):
            if i_t == ss:
                new_char_ss = len(' '.join(sents)) + 1 if sents else 1
                if sents:
                    sents += ['@'] + ['*'] + subj_type + ['*'] + [token]
                else:
                    sents = [' @'] + ['*'] + subj_type + ['*'] + [token]
            if i_t == se:
                if i_t != ss:
                    sents += [token] + ['@']
                else:
                    sents += ['@']
                new_char_se = len(' '.join(sents)) - 1
            if i_t == os:
                new_char_os = len(' '.join(sents)) + 1 if sents else 1
                if sents:
                    sents += ["#"] + ['^'] + obj_type + ['^'] + [token]
                else:
                    sents = [' #'] + ['^'] + obj_type + ['^'] + [token]
            if i_t == oe:
                if i_t != os:
                    sents += [token] + ['#']
                else:
                    sents += ['#']
                new_char_oe = len(' '.join(sents)) - 1
            if i_t != ss and i_t != se and i_t != os and i_t != oe: sents += [token]
        encoding = self.tokenizer.encode_plus(' '.join(sents), max_length=self.max_seq_length, truncation=True)
        new_ss, new_os = encoding.char_to_token(new_char_ss), encoding.char_to_token(new_char_os)
        new_se, new_oe = encoding.char_to_token(new_char_se), encoding.char_to_token(new_char_oe)

        input_ids = encoding['input_ids']
        # input_ids = encoding['input_ids'][:self.max_seq_length]

        # if self.mask:  # 如果需要mask，则是直接替换成mask_token，不能直接换，因为还有实体类型信息
        #     pass

        return input_ids, new_ss, new_se, new_os, new_oe


if __name__ == '__main__':
    test_dataset = json.load(open('../../data/re-datasets/tacred/train-aug.json'))
    # subj_types = json.load(open('data/tacred/subj_types.json'))
    # obj_types = json.load(open('data/tacred/obj_types.json'))

    idx = 0
    # def test_idx(idx):
    for i in range(2):
        d = test_dataset[idx][i]
        ss, se = d['subj_start'], d['subj_end']
        os, oe = d['obj_start'], d['obj_end']

        tokens = d['token']
        tokens = [convert_token(token) for token in tokens]

        tokenizer = AutoTokenizer.from_pretrained('roberta-base')
        em_encoder = TextEncoder.build_text_encoder('typed_entity_marker_punct_new', tokenizer, 512, None, None)
        # new_em_encoder = TextEncoder.build_text_encoder('typed_entity_marker_punct', tokenizer, 512, subj_types, obj_types)

        out1 = em_encoder.mask_encode(tokens, d['subj_type'], d['obj_type'], ss, se, os, oe, mask_rate=0.)
        # out2 = new_em_encoder.encode(tokens, d['subj_type'], d['obj_type'], ss, se, os, oe)

        ids1, ss1, se1, os1, oe1 = out1
        # ids2, ss2, se2, os2, oe2 = out2

        # print(out1)
        # print(f'out1 == out2: {out1 == out2}')
        # print('-' * 100)
        print(tokens)
        print(ids1)
        print(f'out1 sentence: {tokenizer.decode(ids1)}')
        print(f'subj_entity: {tokenizer.decode(ids1[ss1:se1+1])}')
        print(f'obj_entity: {tokenizer.decode(ids1[os1:oe1+1])}')
        # print(f'out2 sentence: {tokenizer.decode(ids2)}')
        # print('-' * 100)
        # print(f'out1 subj: {tokenizer.decode(ids1[ss1:se1 + 1])}, obj: {tokenizer.decode(ids1[os1:oe1 + 1])}')
        # print(f'out2 subj: {tokenizer.decode(ids2[ss2:se2 + 1])}, obj: {tokenizer.decode(ids2[os2:oe2 + 1])}')
