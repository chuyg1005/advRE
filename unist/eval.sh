export CUDA_VISIBLE_DEVICES=1

dataset=$1
model=$2
split=$3
python eval.py --data_dir ../../re-datasets --ckpt_dir ./output/${dataset}/roberta-base-${model} --dataset ${dataset} --split ${split} --mask
