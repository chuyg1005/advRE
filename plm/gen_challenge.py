"""生成数据集的挑战集的代码，需要提前训练好entity-name-only的模型"""
import json
import os
from argparse import ArgumentParser

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dataset', type=str, default='tacred')
    parser.add_argument('--split', type=str, default='test')
    args = parser.parse_args()

    data_dir = '../../re-datasets'
    org_file = os.path.join(data_dir, args.dataset, args.split+'.json')
    org_data = json.load(open(org_file))
    entity_preds = json.load(open('saved_models/' + args.dataset + '/entity-name-only/' + args.split + '-pred.json'))
    context_preds = json.load(open('saved_models/' + args.dataset + '/entity-context-only/' + args.split + '-pred.json'))

    assert len(org_data) == len(entity_preds), '预测结果和原始文件不符！'
    assert len(org_data) == len(context_preds), '预测结果和原始文件不符！'

    filtered_data = []
    for i, (ent_pred, cont_pred)  in enumerate(zip(entity_preds, context_preds)):
        assert ent_pred[0] == cont_pred[0]
        if ent_pred[0] != ent_pred[1] and cont_pred[0] == cont_pred[1]: # 依赖实体名称无法预测正确，但是依赖上下文可以预测正确
            print(ent_pred[0], ent_pred[1])
            filtered_data.append(org_data[i]) # 加入到挑战集

    print(f'剩余数量：{len(filtered_data)} / {len(org_data)}.') # 打印数据条数
    # with open(os.path.join(data_dir, args.dataset, 'splits', args.split+'_challenge.json'), 'w') as f:
    #     json.dump(filtered_data, f)