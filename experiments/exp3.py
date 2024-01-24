import os
import sys
import argparse
import numpy as np
import random
import torch
from torch.utils.data import DataLoader
from spikingjelly.datasets import asl_dvs, cifar10_dvs, dvs128_gesture, n_caltech101, n_mnist

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import models.spiking_resnet as spiking_resnet
import models.sew_resnet as sew_resnet
from assess import rank

_seed_ = 2020
random.seed(2020)
torch.manual_seed(_seed_)
torch.cuda.manual_seed_all(_seed_)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(_seed_)

def parser_args():
    parser = argparse.ArgumentParser(description='transfer SNN')
    # data
    parser.add_argument('--nclasses', default=11, type=int, help='number of classes')
    parser.add_argument('--batch_size', default=128, type=int, help='batch size')
    parser.add_argument('--nworkers', default=16, type=int, help='number of workers')
    # model
    parser.add_argument('--connect_f', default='ADD', type=str, help='spike-element-wise connect function')
    # run
    parser.add_argument('--device_id', default=0, type=int, help='GPU id to use.')
    return parser.parse_args()


def load_data(args):
    dataset = dvs128_gesture.DVS128Gesture(root='/home/haohq/datasets/DVS128Gesture', train=False, data_type='frame', frames_number=8, split_by='time')
    sampler = torch.utils.data.RandomSampler(dataset)
    return DataLoader(dataset, batch_size=args.batch_size, sampler=sampler, num_workers=args.nworkers, pin_memory=True, drop_last=False)

def load_model(model, pretrained_path, args):
    # load weights
    if not os.path.exists(pretrained_path):
        raise FileNotFoundError(pretrained_path)
    else:
        checkpoint = torch.load(pretrained_path)
        if 'model' in checkpoint.keys():
            pretrained_weights = checkpoint['model']
        else:
            pretrained_weights = checkpoint

    # model
    if model in sew_resnet.__dict__:
        model = sew_resnet.__dict__[model](num_classes=11, T=8, connect_f=args.connect_f)
        if pretrained_weights is not None:
            params = model.state_dict()
            for k, v in pretrained_weights.items():
                if k in params:
                    params[k] = v
            model.load_state_dict(params)
            print('load pretrained weights from {}'.format(pretrained_path))
    elif model in spiking_resnet.__dict__:
        model = spiking_resnet.__dict__[model](num_classes=11, T=8)
        if pretrained_weights is not None:
            params = model.state_dict()
            for k, v in pretrained_weights.items():
                if k in params:
                    params[k] = v
            model.load_state_dict(params)
            print('load pretrained weights from {}'.format(pretrained_path))
    else:
        raise NotImplementedError(model)

    return model

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

pretrained_weights = [
    '/home/haohq/SNN-Trans-Assess-main/weights/sew18_checkpoint_319.pth',
    '/home/haohq/SNN-Trans-Assess-main/weights/sew34_checkpoint_319.pth',
    '/home/haohq/SNN-Trans-Assess-main/weights/sew50_checkpoint_319.pth',
    '/home/haohq/SNN-Trans-Assess-main/weights/sew101_checkpoint_319.pth',
    '/home/haohq/SNN-Trans-Assess-main/weights/sew152_checkpoint_319.pth',
    '/home/haohq/SNN-Trans-Assess-main/weights/spiking_resnet_18_checkpoint_319.pth',
    '/home/haohq/SNN-Trans-Assess-main/weights/spiking_resnet_34_checkpoint_319.pth',
    '/home/haohq/SNN-Trans-Assess-main/weights/spiking_resnet_50_checkpoint_319.pth',
]

if __name__ == '__main__':
    args = parser_args()
    models = []
    for model, pretrained_path in zip(model_names, pretrained_weights):
        models.append(load_model(model, pretrained_path, args))
    
    dataloader = load_data(args)
    scores = rank(models, dataloader, mode='ApproxME')
    print(scores)
