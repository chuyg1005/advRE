# bash train/run.sh 0 tacred baseline
export CUDA_VISIBLE_DEVICES=$1
# 使用我们的方法训练模型
dataset=$2
# model_name=$3
# model_name=${4-roberta-base} # 默认选择roberta-base模型
# input_format=entity_name_only
python train_tacred.py \
    --data_dir ../../re-datasets/${dataset} \
    --model_name_or_path roberta-base \
    --train_batch_size 64 \
    --input_format entity_type_only \
    --run_name entity-type-only \
    --train_name train-gpt2 \
    --train_mode baseline

# python train_tacred.py \
#     --data_dir ../../re-datasets/${dataset} \
#     --model_name_or_path roberta-base \
#     --train_batch_size 64 \
#     --input_format entity_name_only \
#     --run_name entity-name-only \
#     --train_name train-gpt2 \
#     --train_mode baseline
# # --seed 0 \

# python train_tacred.py \
#     --data_dir ../../re-datasets/${dataset} \
#     --model_name_or_path roberta-base \
#     --train_batch_size 64 \
#     --input_format entity_mask_new \
#     --run_name entity-context-only \
#     --train_name train-gpt2 \
#     --train_mode baseline

# python train_tacred.py \
#     --data_dir ../../re-datasets/${dataset} \
#     --model_name_or_path roberta-base \
#     --train_batch_size 64 \
#     --input_format entity_marker_punct_new \
#     --run_name entity-context-and-name \
#     --train_name train-gpt2 \
#     --train_mode baseline
