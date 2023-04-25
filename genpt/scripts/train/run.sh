# bash train/run.sh 0 tacred baseline
export CUDA_VISIBLE_DEVICES=$1

dataset=$2
train_mode=$3 # baseline / data-aug / ours
model_name=t5-base

python3 code/run_prompt.py \
    --data_name ${dataset} \
    --data_dir ../../re-datasets/${dataset} \
    --output_dir ./results/${dataset}/${model_name}-${train_mode} \
    --model_type T5 \
    --model_name_or_path ${model_name} \
    --per_gpu_train_batch_size 4 \
    --per_gpu_eval_batch_size 32 \
    --gradient_accumulation_steps 1 \
    --max_seq_length 512 \
    --max_ent_type_length 7 \
    --max_label_length 9 \
    --warmup_steps 500 \
    --learning_rate 3e-5 \
    --learning_rate_for_new_token 1e-5 \
    --num_train_epochs 5 \
    --rel2id_dir ./data/${dataset}/rela2id.json \
    --max_grad_norm 2 \
    --train_mode ${train_mode} \
    --train_name train-gpt2 \
    --seed 0
