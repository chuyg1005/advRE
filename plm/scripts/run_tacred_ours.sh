export CUDA_VISIBLE_DEVICES=1;
# 使用我们的方法训练模型
dataset=tacred;
model_name=roberta-base;
python train_tacred.py --data_dir ../../re-datasets/${dataset} --model_name_or_path ${model_name} --train_batch_size 64 --input_format typed_entity_marker_punct_new --run_name roberta-base-ours-gpt-filtered --train_name train-gpt2 --train-mode ours --seed 42;