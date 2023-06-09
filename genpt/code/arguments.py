import argparse

import torch
import transformers
from transformers import (AlbertConfig, AlbertForMaskedLM, AlbertTokenizer,
                          AutoTokenizer, BartConfig,
                          BartForConditionalGeneration, BartTokenizer,
                          BertConfig, BertForMaskedLM, BertTokenizer,
                          RobertaConfig, RobertaForMaskedLM, RobertaTokenizer,
                          T5Config, T5ForConditionalGeneration, T5Tokenizer)

# from transformers.modeling_bert import BertModel, BertOnlyMLMHead
# from transformers.modeling_roberta import RobertaModel, RobertaLMHead

_GLOBAL_ARGS = None

_MODEL_CLASSES = {
    'roberta': {
        'config': RobertaConfig,
        'tokenizer': RobertaTokenizer,
        'model': RobertaForMaskedLM,
    },
    'T5': {
        'config': T5Config,
        # 'tokenizer': T5Tokenizer,
        'tokenizer': AutoTokenizer, # 切换为autoTokenizer
        'model': T5ForConditionalGeneration,
    },
    'Bart': {
        'config': BartConfig,
        'tokenizer': BartTokenizer,
        'model': BartForConditionalGeneration,
    },
}


def get_embedding_layer(args, model):
    if 'roberta' in args.model_type:
        embeddings = model.roberta.get_input_embeddings()
    elif 'bert' in args.model_type:
        embeddings = model.bert.get_input_embeddings()
    elif 'gpt' in args.model_type:
        embeddings = model.base_model.get_input_embeddings()
    elif 'megatron' in args.model_type:
        embeddings = model.decoder.embed_tokens
    elif 'T5' in args.model_type:
        embeddings = model.encoder.embed_tokens
    elif 'Bart' in args.model_type:
        embeddings = model.model.get_input_embeddings()
    else:
        raise NotImplementedError()
    return embeddings


def get_args_parser():

    parser = argparse.ArgumentParser(description="Command line interface for Relation Extraction.")

    # Required parameters

    parser.add_argument("--data_name", type=str, required=True, choices=['tacred', 'tacrev', 'retacred', 'wiki80'],
                        help="The name of data.")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="The input data dir. Should contain the data files for the task.")
    parser.add_argument("--model_type", default="bart", type=str, required=True, choices=_MODEL_CLASSES.keys(),
                        help="The type of the pretrained language model to use")
    parser.add_argument("--model_name_or_path", default="albert-xxlarge-v2", type=str, required=True,
                        help="Path to the pre-trained model or shortcut name")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written")
    parser.add_argument("--rel2id_dir", default=None, type=str, required=True,
                        help="The rel2id data dir.")

    parser.add_argument("--k", default=0, type=int,
                        help="The number of labeled data per class.")
    parser.add_argument("--data_seed", default=0, type=int,
                        help="The seed of sampled few-shot data.")
    parser.add_argument("--pseudo_token", default="[PROMPT]", type=str,
                        help="The pseudo token of soft prompt")
    parser.add_argument("--prompt_lens", default=[3, 3, 3], type=list,
                        help="The output directory where the model predictions and checkpoints will be written")
    parser.add_argument("--lstm_dropout", type=float, default=0.0)

    parser.add_argument("--max_seq_length", default=256, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--max_ent_type_length", default=4, type=int,
                        help="The maximum total entity type length")
    parser.add_argument("--max_label_length", default=6, type=int,
                        help="The maximum total label length.")
    parser.add_argument("--per_gpu_train_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for PET training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for PET evaluation.")
    # 累积梯度
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass in PET.")
    parser.add_argument("--num_train_epochs", default=3, type=float,
                        help="Total number of training epochs to perform in PET.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform in PET. Override num_train_epochs.")

    # Other optional parameters
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Where to store the pre-trained models downloaded from S3.")
    parser.add_argument("--learning_rate", default=1e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--learning_rate_for_new_token", default=1e-5, type=float,
                        help="The initial learning rate for Adam.")

    parser.add_argument("--weight_decay", default=1e-2, type=float,
                        help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-6, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")


    # * 添加我们的配置
    # * 0: baseline
    # * 1: ours
    # * 2: ablation
    parser.add_argument('--train_mode', type=str, default='baseline', help='baseline / data-aug / ours')
    parser.add_argument("--mode", type=int, default=0, help="训练模型（0: baseline, 1: ours, 2: ablation).")
    parser.add_argument("--train_name", type=str, default="train", help="训练集的文件名")
    parser.add_argument('--mask_entity', action='store_true', help='mask_entity')

    # * 判断是否需要进行训练
    parser.add_argument("--eval_only", action="store_true", help="用来评价模型")
    parser.add_argument("--eval_name", default='test', type=str, help="评价数据文件的名称")

    args = parser.parse_args()
    args.n_gpu = torch.cuda.device_count()

    global _GLOBAL_ARGS
    _GLOBAL_ARGS = args
    return args

def get_args():
    return _GLOBAL_ARGS

def get_model_classes():
    return _MODEL_CLASSES
