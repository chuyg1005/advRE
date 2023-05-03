# bash eval.sh 0 tacred baseline test
# bash eval.sh 0 tacred baseline test_rev
# bash eval.sh 0 retacred baseline test
export CUDA_VISIBLE_DEVICES=$1
model_name=bart-base-$3
dataset=$2
split=$4
data_name=$dataset
# 如果split是test_rev，则切换data_name为tacrev
if [ $split = 'test_rev' ]; then
    data_name=tacrev
fi

# 不mask entity name
python code/run_prompt.py \
    --data_name ${data_name} \
    --data_dir ../../re-datasets/${dataset} \
    --output_dir ./results/${dataset}/${model_name} \
    --model_type Bart \
    --model_name_or_path facebook/bart-base \
    --per_gpu_eval_batch_size 32 \
    --max_seq_length 512 \
    --max_ent_type_length 4 \
    --max_label_length 6 \
    --learning_rate 3e-5 \
    --learning_rate_for_new_token 1e-5 \
    --rel2id_dir ./data/${dataset}/rela2id.json \
    --eval_only \
    --eval_name ${split} \
    --mask_entity

# mask住entity name
# python code/run_prompt.py \
    #--data_name ${data_name} \
    # --data_dir ../../re-datasets/${dataset} \
    # --output_dir ./results/${dataset}/${model_name} \
    # --model_type Bart \
    # --model_name_or_path facebook/bart-base \
    # --per_gpu_eval_batch_size 32 \
    # --max_seq_length 512 \
    # --max_ent_type_length 7 \
    # --max_label_length 9 \
    # --learning_rate 3e-5 \
    # --learning_rate_for_new_token 1e-5 \
    # --rel2id_dir ./data/${dataset}/rela2id.json \
    # --eval_only \
    # --eval_name ${split} #\
#    --mask_entity
