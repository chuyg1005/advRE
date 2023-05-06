# bash train/run.sh 0 tacred baseline
export CUDA_VISIBLE_DEVICES=$1
# 使用我们的方法训练模型
dataset=$2
# model_name=$3
# model_name=roberta-base
# train_mode=$3
train_mode=$3           # 不带有3sigma规则筛选
model_name=roberta-base # 默认选择roberta-base模型
python train_tacred.py \
    --data_dir ../../re-datasets/${dataset} \
    --model_name_or_path ${model_name} \
    --train_batch_size 64 \
    --input_format typed_entity_marker_punct_new \
    --run_name ${model_name}-${train_mode} \
    --train_name train-aug \
    --train_mode ${train_mode}
