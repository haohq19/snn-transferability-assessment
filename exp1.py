# calculate LogME, NCE and LEEP with cached features and outputs

import os
import argparse
from engines.assess import assess_with_cache

def parser_args():
    parser = argparse.ArgumentParser(description='assess SNN with cache')
    # data
    parser.add_argument('--dataset', default=None, type=str, help='dataset')
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
        score1, score2, score3 = assess_with_cache(cache_dir=cache_dir)
        print('NCE: {:.8f}, LEEP: {:.8f}, logME: {:.8f}'.format(score1, score2, score3))