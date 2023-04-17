export CUDA_VISIBLE_DEVICES=2
# 跑tacred
for model_name in 'ours'; do
     python code/run_prompt.py \
          --data_name tacred \
          --data_dir ../../re-datasets/tacred \
          --output_dir ./results/tacred_${model_name} \
          --model_type T5 \
          --model_name_or_path t5-base \
          --per_gpu_eval_batch_size 32 \
          --max_seq_length 512 \
          --max_ent_type_length 7 \
          --max_label_length 9 \
          --learning_rate 3e-5 \
          --learning_rate_for_new_token 1e-5 \
          --rel2id_dir ./data/tacred/rela2id.json \
          --eval_only \
          --eval_name test
done
    #  --per_gpu_train_batch_size 4 \
    #  --gradient_accumulation_steps 1 \
    #  --warmup_steps 500 \
    #  --num_train_epochs 5 \
    #  --max_grad_norm 2 \