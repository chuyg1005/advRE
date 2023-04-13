export CUDA_VISIBLE_DEVICES=2;
dataset=retacred;
python train_tacred.py --data_dir ../re-datasets/${dataset} --model_name_or_path roberta-base --train_batch_size 64 --input_format typed_entity_marker_punct_new --run_name typed_entity_name_baseline --train_name train-aug --use_baseline --seed 42;