python run.py \
--data_dir ./data \
--output_dir output/fewrel/roberta-base \
--train_tasks fewrel \
--eval_tasks fewrel \
--do_train \
--eval_during_training \
--logging_epochs 10 \
--save_epochs 10 \
--num_train_epochs 50 \
--per_gpu_train_batch_size 64 \
--model_name_or_path roberta-base \