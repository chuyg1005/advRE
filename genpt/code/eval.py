"""需要提前保存好预测的结果，然后再计算"""
import json
import os
from argparse import ArgumentParser

from utils import f1_score


def load_data(path):
    return json.load(open(path))

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dataset', required=True, type=str, help='数据集名字')
    parser.add_argument('--model_name', required=True, type=str, help='模型名字')
    parser.add_argument('--model_type', default='t5-base', type=str, help='模型类型')
    # parser.add_argument('--eval_result_path', required=True, type=str, help="评价结果文件eval_results.json的路径")
    parser.add_argument('--data_dir', default='../../re-datasets', type=str, help='splits数据集路径')
    # parser.add_argument('--rela2id_path', required=True, type=str, help='rela2id.json的路径')
    parser.add_argument('--split', required=True, type=str, help='数据子集的名字')
    parser.add_argument('--mask', action='store_true')

    args = parser.parse_args()
    if args.split.startswith("test_rev"):
        pred_file = 'test_rev'
    else:
        pred_file = 'test'
    if args.mask: pred_file += '_mask'
    # pred_file = args.split + '_mask' if args.mask else args.split 

    eval_result = load_data(os.path.join('results', args.dataset, f'{args.model_type}-{args.model_name}', pred_file + '.json'))
    eval_data = load_data(os.path.join(args.data_dir, args.dataset, 'splits', args.split + '.json'))
    rela2id = load_data(os.path.join('data', args.dataset, 'rela2id.json'))

    # print('load data success.')
    indices = list(map(lambda item: item['test_idx'], eval_data)) # 反向索引
    result = [eval_result[idx] for idx in indices]

    pred = [pair[0] for pair in result]
    label = [pair[1] for pair in result]

    mi_f1, ma_f1 = f1_score(pred, label, len(rela2id), rela2id['no_relation'])

    print(f'[{args.split}] mi_f1: {100 * mi_f1:.2f}, ma_f1: {100*ma_f1:.2f}.')