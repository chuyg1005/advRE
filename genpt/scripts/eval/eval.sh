# for split in 'test' 'test_two_new_entity' 'test_one_new_entity' 'test_new_subj_entity' 'test_new_obj_entity' 'test_two_old_entity' 'test_two_old_entity_new_pair' 'test_two_old_entity_old_pair' 'test_two_old_entity_old_pair_new_rela' 'test_two_old_entity_old_pair_old_rela'; do
#     python code/eval.py \
#         --data_dir  ../../re-datasets/tacred \
#         --eval_result_path ./results/tacred_ours/test.json \
#         --rela2id_path ./data/tacred/rela2id.json \
#         --split ${split}
# done

for split in 'test_rev' 'test_rev_two_new_entity' 'test_rev_one_new_entity' 'test_rev_new_subj_entity' 'test_rev_new_obj_entity' 'test_rev_two_old_entity' 'test_rev_two_old_entity_new_pair' 'test_rev_two_old_entity_old_pair' 'test_rev_two_old_entity_old_pair_new_rela' 'test_rev_two_old_entity_old_pair_old_rela'; do
    python code/eval.py \
        --data_dir  ../../re-datasets/tacred \
        --eval_result_path ./results/tacred_ablation/test_rev.json \
        --rela2id_path ./data/tacred/rela2id.json \
        --split ${split}
done