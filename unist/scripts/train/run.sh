# bash train/run.sh 0 tacred baseline
export CUDA_VISIBLE_DEVICES=$1
dataset=$2
train_mode=$3
# model_name=roberta-base
model_name=${4-roberta-base} # 默认选择roberta-base模型
python run.py \
    --data_dir ../../re-datasets/${dataset} \
    --output_dir output/${dataset}/${model_name}-${train_mode} \
    --train_name train-gpt2 \
    --train_tasks ${dataset} \
    --eval_tasks ${dataset} \
    --do_train \
    --eval_during_training \
    --logging_epochs 5 \
    --save_epochs 5 \
    --num_train_epochs 100 \
    --max_sent_length 512 \
    --per_gpu_train_batch_size 64 \
    --model_name_or_path ${model_name} \
    --train_mode ${train_mode}
# --no_task_desc \
# --learning_rate 3e-5 \
# 设置max_sent_length防止实体超出句子范围
# --use_pseudo \
# --aug_weight 0.5 \
