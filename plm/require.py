import torch
import time
import os
import argparse
import shutil
import sys


def parse_args():
    parser = argparse.ArgumentParser(description='Matrix multiplication')
    # parser.add_argument('--gpus', help='gpu amount', required=True, type=int)
    parser.add_argument('--size', help='matrix size', default=20000, type=int)
    parser.add_argument('--interval', help='sleep interval', default=0.01, type=float)
    args = parser.parse_args()
    return args


def matrix_multiplication(args):
    # a_list, b_list, result = [], [], []
    size = (args.size, args.size)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    a = torch.rand(size).to(device)
    b = torch.rand(size).to(device)
    c = torch.rand(size).to(device)

    while True:
        with torch.no_grad():
            c = a * b
        time.sleep(args.interval)


if __name__ == "__main__":
    # usage: python matrix_multiplication_gpus.py --size 20000 --gpus 2 --interval 0.01
    args = parse_args()
    matrix_multiplication(args)
