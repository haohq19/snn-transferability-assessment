# transfer pretrained SNN on event-based datasets

import os
import argparse
import csv
import numpy as np
import random
import yaml
import torch
import torch.nn as nn
import models.spiking_resnet_event as spiking_resnet
import models.sew_resnet_event as sew_resnet
from models.linear_probe import LinearProbe
from utils.data import get_event_data_loader
from engines.transfer import cache_representations, get_data_loader_from_cached_representations, train, test


_seed_ = 2024
random.seed(2024)
torch.manual_seed(_seed_)
torch.cuda.manual_seed_all(_seed_)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(_seed_)

def parser_args():
    parser = argparse.ArgumentParser(description='transfer SNN on event-based datasets')
    # data
    parser.add_argument('--dataset', default='dvs128_gesture', type=str, help='dataset')
    parser.add_argument('--nsteps', default=8, type=int, help='number of time steps')
    parser.add_argument('--num_classes', default=11, type=int, help='number of classes')
    parser.add_argument('--batch_size', default=1024, type=int, help='batch size')
    # model
    # parser.add_argument('--model', default='sew_resnet18', type=str, help='model type')
    parser.add_argument('--connect_f', default='ADD', type=str, help='spike-element-wise connect function')
    # run
    parser.add_argument('--device_id', default=0, type=int, help='GPU id to use.')
    parser.add_argument('--nepochs', default=100, type=int, help='number of epochs')
    parser.add_argument('--nworkers', default=8, type=int, help='number of workers')
    # parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    # parser.add_argument('--weight_decay', default=0, type=float, help='weight decay')
    parser.add_argument('--pt_dir', default='weights', help='path to pretrained weights')
    parser.add_argument('--output_dir', default='outputs/transfer', help='path where to save')
    return parser.parse_args()


def _get_output_dir(args):

    output_dir = os.path.join(args.output_dir, args.dataset, args.model)
    output_dir = os.path.join(output_dir, f'lr{args.lr}_wd{args.weight_decay}_ep{args.nepochs}_T{args.nsteps}_cnf{args.connect_f}')
    
    return output_dir


def _get_model(args):
    # load pretrained weights
    pt_path = os.path.join(args.pt_dir, args.model + '.pth')
    if not os.path.exists(pt_path):
        raise FileNotFoundError(pt_path)
    else:
        checkpoint = torch.load(pt_path)
        if 'model' in checkpoint.keys():
            pt_weights = checkpoint['model']
        else:
            pt_weights = checkpoint

    # model
    if args.model in sew_resnet.__dict__:
        model = sew_resnet.__dict__[args.model](num_classes=1000, T=args.nsteps, connect_f=args.connect_f)
        if pt_weights is not None:
            state_dict = {k.replace('module.', ''): v for k, v in pt_weights.items()}
            model.load_state_dict(state_dict)
            print('Load pretrained weights from [{}]'.format(pt_path))
    elif args.model in spiking_resnet.__dict__:
        model = spiking_resnet.__dict__[args.model](num_classes=1000, T=args.nsteps)
        if pt_weights is not None:
            state_dict = {k.replace('module.', ''): v for k, v in pt_weights.items()}
            model.load_state_dict(state_dict)
            print('Load pretrained weights from [{}]'.format(pt_path))
    else:
        raise NotImplementedError(args.model)
    
    for param in model.parameters():
        param.requires_grad = False

    return model


def _get_num_dims(args):
    num_dims_dict = {
        'sew_resnet18': 512,
        'sew_resnet34': 512,
        'sew_resnet50': 2048,
        'sew_resnet101': 2048,
        'sew_resnet152': 2048,
        'spiking_resnet18': 512,
        'spiking_resnet34': 512,
        'spiking_resnet50': 2048,
    }
    return num_dims_dict[args.model]


def main(args):
    
    # device
    torch.cuda.set_device(args.device_id)

    # criterion
    criterion = nn.MSELoss()
    
    # output dir
    output_dir = _get_output_dir(args)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print('Mkdir [{}]'.format(output_dir))
    if not os.path.exists(os.path.join(output_dir, 'checkpoints')):
        os.makedirs(os.path.join(output_dir, 'checkpoints'))
        print('Mkdir [{}]'.format(os.path.join(output_dir, 'checkpoints')))

    # model
    model = _get_model(args)
    model.cuda()
    
    # cache representations
    cache_dir = os.path.join(args.output_dir, args.dataset, args.model, 'cache')  # output_dir/dataset/model/cache
    args.cache_dir = cache_dir
    train_loader, valid_loader, test_loader = get_event_data_loader(args)

    cache_representations(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        test_loader=test_loader,
        cache_dir=cache_dir,
    )
    train_loader, valid_loader, test_loader = get_data_loader_from_cached_representations(args)

    # linear_probe
    args.num_dims = _get_num_dims(args)
    linear_probe = LinearProbe(args.num_dims, args.num_classes)
    linear_probe.cuda()

    # run
    params = filter(lambda p: p.requires_grad, linear_probe.parameters())
    optimizer = torch.optim.Adam(params, lr=args.lr,  weight_decay=args.weight_decay)
    
    # print and save args
    print(args)
    with open(os.path.join(output_dir, 'config.yaml'), 'w') as f:
        yaml.dump(vars(args), f, default_flow_style=False)


    best_model, best_valid_acc = train(
        model=linear_probe,
        criterion=criterion,
        optimizer=optimizer,
        train_loader=train_loader,
        valid_loader=valid_loader,
        num_classes=args.num_classes,
        nepochs=args.nepochs,
        output_dir=output_dir,
    )

    test_acc = test(
        model=best_model,
        test_loader=test_loader,
    )

    # save results
    result_path = os.path.join(args.output_dir, args.dataset, 'results.csv')  # output_dir/dataset/results.csv
    with open(result_path, 'a') as f:
        writer = csv.writer(f)
        data = [[args.model, args.lr, args.weight_decay, best_valid_acc, test_acc],]
        writer.writerows(data)
    

if __name__ == '__main__':
    models = ['sew_resnet152', 'sew_resnet101', 'sew_resnet50', 'sew_resnet34', 'sew_resnet18', 'spiking_resnet50', 'spiking_resnet34', 'spiking_resnet18']
    learning_rates = [1e-2, 3e-3, 1e-3, 3e-4, 1e-4]
    weight_decays = [1e-5, 3e-6, 1e-6, 3e-7, 1e-7]
    args = parser_args()
    for model in models:
        for lr in learning_rates:
            for weight_decay in weight_decays:
                args.model = model
                args.lr = lr
                args.weight_decay = weight_decay
                main(args)

'''
python transfer_event.py
'''