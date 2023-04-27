dataset=$1
model_name=$2
ckpt_name=$3
python error_analysis.py --ckpt_dir saved_models/${dataset}/${model_name} --dataset test_two_old_entity_old_pair --model_name ${ckpt_name} --eval_data_dir ../../re-datasets/${dataset}
