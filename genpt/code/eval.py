import os
import json
from argparse import ArgumentParser
from utils import f1_score

def load_data(path):
    return json.load(open(path))

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--eval_result_path', required=True, type=str, help="评价结果文件eval_results.json的路径")
    parser.add_argument('--data_dir', required=True, type=str, help='splits数据集路径')
    parser.add_argument('--rela2id_path', required=True, type=str, help='rela2id.json的路径')
    parser.add_argument('--split', required=True, type=str, help='数据子集的名字')

    args = parser.parse_args()

    eval_result = load_data(args.eval_result_path)
    eval_data = load_data(os.path.join(args.data_dir, 'splits', args.split + '.json'))
    rela2id = load_data(args.rela2id_path)

    # print('load data success.')
    indices = list(map(lambda item: item['test_idx'], eval_data))
    result = [eval_result[idx] for idx in indices]

    pred = [pair[0] for pair in result]
    label = [pair[1] for pair in result]

    mi_f1, ma_f1 = f1_score(pred, label, len(rela2id), rela2id['no_relation'])

    print(f'[{args.split}] mi_f1: {100 * mi_f1:.2f}.')