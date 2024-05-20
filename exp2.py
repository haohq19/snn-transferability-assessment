# compare logME and ApproxME

import os
import argparse
from engines.assess import compare_logme_and_approxme_with_cache

def parser_args():
    parser = argparse.ArgumentParser(description='assess SNN with cache')
    # data
    parser.add_argument('--dataset', default='cifar10_dvs', type=str, help='dataset')
    parser.add_argument('--output_dir', default='outputs/transfer', type=str, help='output directory')
    return parser.parse_args()


model_names = [
    'sew_resnet18',
    'sew_resnet34',
    'sew_resnet50',
    'sew_resnet101',
    'sew_resnet152',
    'spiking_resnet18',
    'spiking_resnet34',
    'spiking_resnet50',
]

if __name__ == '__main__':
    args = parser_args()

    score1s = []
    score2s = []
    score3s = []
    for model in model_names:
        cache_dir = os.path.join(args.output_dir, args.dataset, model, 'cache')
        score1, score2, iter1, iter2 = compare_logme_and_approxme_with_cache(cache_dir=cache_dir)
        print('logME: {:.4f}, {:.2f}, ApproxME: {:.4f}, {:.2f}'.format(score1, iter1, score2, iter2))