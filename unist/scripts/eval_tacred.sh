export CUDA_VISIBLE_DEVICES=0
type=ablation
# dataset=retacred
dataset=retacred
split=test
# for split in 'test' 'test_two_new_entity' 'test_one_new_entity' 'test_new_subj_entity' 'test_new_obj_entity' 'test_two_old_entity' 'test_two_old_entity_new_pair' 'test_two_old_entity_old_pair' 'test_two_old_entity_old_pair_new_rela' 'test_two_old_entity_old_pair_old_rela'; do
# python run.py --data_dir ../../data/re-datasets/${dataset} --train_tasks ${dataset} --eval_tasks ${dataset} --do_eval --eval_dir ./output/${dataset}-${type}/roberta-base/checkpoint-90 --eval_name ${split} --eval_only
python run.py --data_dir ../../re-datasets/${dataset} --train_tasks ${dataset} --eval_tasks ${dataset} --do_eval --eval_dir ./output/${dataset}-${type}/roberta-base --eval_name ${split} --eval_only --mask_entity
# done

# for split in 'test_rev' 'test_rev_two_new_entity' 'test_rev_one_new_entity' 'test_rev_new_subj_entity' 'test_rev_new_obj_entity' 'test_rev_two_old_entity' 'test_rev_two_old_entity_new_pair' 'test_rev_two_old_entity_old_pair' 'test_rev_two_old_entity_old_pair_new_rela' 'test_rev_two_old_entity_old_pair_old_rela'; do
#     python run.py --data_dir ../../data/re-datasets/${dataset} --train_tasks ${dataset} --eval_tasks ${dataset} --do_eval --eval_dir ./output/${dataset}-${type}/roberta-base/checkpoint-90 --eval_name ${split} --eval_only
#     # python run.py --data_dir ../../data/re-datasets/${dataset} --train_tasks ${dataset} --eval_tasks ${dataset} --do_eval --eval_dir ./output/${dataset}-${type}/roberta-base --eval_name ${split} --eval_only
# done
