"""从训练集中统计生成一些必要的信息:
1) 关系类型；
2）实体类型；
3）实体类型和关系的对应"""
import json
import os
from argparse import ArgumentParser


def parse_item(item):
    return item['subj_type'], item['obj_type'], item['relation']


def gen_constants(data_dir, save_dir=None):
    if save_dir is None: 
        save_dir = os.path.join(data_dir, 'constants')
        os.makedirs(save_dir, exist_ok=True)
    train_data = []
    for filename in ['train.json', 'dev.json', 'test.json']:
        train_data += json.load(open(os.path.join(data_dir, filename)))
    relas = set()
    subj_types = set()
    obj_types = set()
    entity_types = set()
    divider = '||'
    offset = 100
    entity_type_rela = {}

    for item in train_data:
        subj_type, obj_type, rela = parse_item(item)

        relas.add(rela)
        subj_types.add(subj_type)
        obj_types.add(obj_type)

        key = subj_type + divider + obj_type
        if key not in entity_type_rela: entity_type_rela[key] = set()
        entity_type_rela[key].add(rela)

    entity_types = subj_types | obj_types  # 求并集

    # rela2id, entity_type2id
    rela2id = dict(zip(relas, range(len(relas))))
    entity_type2id = dict(zip(entity_types, range(len(entity_types))))

    entity_type_rela2id = {}
    for entity_type_pair, relas in entity_type_rela.items():
        subj_type, obj_type = entity_type_pair.split(divider)
        subj_type_id, obj_type_id = entity_type2id[subj_type], entity_type2id[obj_type]
        entity_type_rela[entity_type_pair] = list(relas)  # 转换为列表，集合无法序列化
        rela_ids = sorted([rela2id[rela] for rela in relas])

        key = subj_type_id * offset + obj_type_id
        entity_type_rela2id[key] = rela_ids

    with open(os.path.join(save_dir, 'rela2id.json'), 'w') as f:
        json.dump(rela2id, f)
    with open(os.path.join(save_dir, 'entity_type2id.json'), 'w') as f:
        json.dump(entity_type2id, f)
    with open(os.path.join(save_dir, 'entity_type_rela2id.json'), 'w') as f:
        json.dump(entity_type_rela2id, f)
    with open(os.path.join(save_dir, 'entity_type_rela.json'), 'w') as f:
        json.dump(entity_type_rela, f)

    with open(os.path.join(save_dir, 'subj_types.json'), 'w') as f:
        json.dump(list(subj_types), f)
    with open(os.path.join(save_dir, 'obj_types.json'), 'w') as f:
        json.dump(list(obj_types), f)


def load_constants(save_dir):
    pass
    # save_dir = os.path.join(save_dir, 'constants')
    # with open(os.path.join(save_dir, 'rela2id.json')) as f:
    #     rela2id = json.load(f)

    # with open(os.path.join(save_dir, 'entity_type2id.json')) as f:
    #     entity_type2id = json.load(f)

    # with open(os.path.join(save_dir, 'entity_type_rela2id.json')) as f:
    #     entity_type_rela2id = json.load(f)

    # with open(os.path.join(save_dir, 'entity_type_rela.json')) as f:
    #     entity_type_rela = json.load(f)

    # with open(os.path.join(save_dir, 'subj_types.json')) as f:
    #     subj_types = json.load(f)

    # with open(os.path.join(save_dir, 'obj_types.json')) as f:
    #     obj_types = json.load(f)

    # return rela2id, entity_type2id, entity_type_rela2id, entity_type_rela, subj_types, obj_types


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    args = parser.parse_args()

    gen_constants(args.data_dir)
