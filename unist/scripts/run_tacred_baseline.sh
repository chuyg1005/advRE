# python run.py \
# --data_dir ./data \
# --output_dir output/tacred/roberta-base \
# --train_tasks tacred \
# --eval_tasks tacred \
# --do_train \
# --eval_during_training \
# --logging_epochs 10 \
# --save_epochs 10 \
# --num_train_epochs 100 \
# --per_gpu_train_batch_size 64 \
# --model_name_or_path roberta-base \
export CUDA_VISIBLE_DEVICES=0;
python run.py \
--data_dir ../../re-datasets/tacred \
--output_dir output/tacred-baseline-gradscaled/roberta-base \
--train_name train \
--train_tasks tacred \
--eval_tasks tacred \
--do_train \
--eval_during_training \
--logging_epochs 5 \
--save_epochs 5 \
--num_train_epochs 100 \
--max_sent_length 512 \
--per_gpu_train_batch_size 64 \
--model_name_or_path roberta-base 
# --no_task_desc \
# --learning_rate 3e-5 \
# 设置max_sent_length防止实体超出句子范围
# --use_pseudo \
# --aug_weight 0.5 \