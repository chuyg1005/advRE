export CUDA_VISIBLE_DEVICES=1
# input_format=$1
# dataset=$2
# eval_name=$3
# python eval.py --model_name_or_path roberta-base --input_format ${input_format} --ckpt_dir checkpoints/${dataset}/${input_format}/best-model.ckpt --data_dir data/${dataset} --dataset ${eval_name}
# for split in 'test' 'test_two_new_entity' 'test_new_subj_entity' 'test_new_obj_entity' 'test_two_old_entity_new_pair' 'test_two_old_entity_old_pair_new_rela' 'test_two_old_entity_old_pair_old_rela'; do
# for split in 'test_two_old_entity_new_pair' 'test_two_old_entity_old_pair_new_rela' 'test_two_old_entity_old_pair_old_rela'; do
#     python eval.py --ckpt_dir saved_models/tacred/typed_entity_name_ours_new --eval_data_dir ../../re-datasets/tacred --dataset ${split};
# done
python eval.py --ckpt_dir saved_models/tacred/roberta-base-baseline --eval_data_dir ../../re-datasets/tacred --dataset test

# for dataset in 'semeval' 'wiki80' 'tacred' 'retacred'; do
#     for filename in `ls ~/data/re-datasets/${dataset}/splits`; do
#         split=${filename%.json}
#         if [ "$split" != 'infos' ]; then 
#             for format in 'entity_name' 'entity_type' 'typed_entity_name'; do
#                 python eval.py --ckpt_dir checkpoints/${dataset}/${format} --dataset $split --eval_data_dir ~/data/re-datasets/${dataset}
#             done
#         fi
#         # if split != 'infos'
#     done
#     # for input_format in 'entity_name' 'entity_type' 'entity_typed_name'; do

#     # done
# done
