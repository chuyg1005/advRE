export CUDA_VISIBLE_DEVICES=3;
# 使用我们的方法训练模型
dataset=tacred;
python train_tacred.py --data_dir ../../re-datasets/${dataset} --model_name_or_path roberta-base --train_batch_size 64 --input_format typed_entity_marker_punct_new --run_name typed_entity_name_ours_new --train_name train-test --seed 42;