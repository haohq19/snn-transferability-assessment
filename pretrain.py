# pretrained SNN on ES-ImageNet

import os
import argparse
import numpy as np
import random
import yaml
import glob
import torch
import torch.nn as nn
import models.spiking_resnet_event as spiking_resnet
import models.sew_resnet_event as sew_resnet
import datasets.es_imagenet as es_imagenet
from torch.utils.data import DataLoader
from utils.distributed import is_master, init_dist
from engines.pretrain import train

_seed_ = 2020
random.seed(2020)
torch.manual_seed(_seed_)
torch.cuda.manual_seed_all(_seed_)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(_seed_)

def parser_args():
    parser = argparse.ArgumentParser(description='pretrain SNN')
    # data
    parser.add_argument('--dataset', default='es_imagenet', type=str, help='dataset')
    parser.add_argument('--root', default=None, type=str, help='path to dataset')
    parser.add_argument('--nsteps', default=8, type=int, help='number of time steps')
    parser.add_argument('--num_classes', default=1000, type=int, help='number of classes')
    parser.add_argument('--batch_size', default=60, type=int, help='batch size')
    # model
    parser.add_argument('--model', default='sew_resnet18', type=str, help='model type')
    parser.add_argument('--connect_f', default='ADD', type=str, help='spike-element-wise connect function')
    # run
    parser.add_argument('--device_id', default=0, type=int, help='GPU id to use, invalid when distributed training')
    parser.add_argument('--nepochs', default=10, type=int, help='number of epochs')
    parser.add_argument('--nworkers', default=32, type=int, help='number of workers')
    parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
    parser.add_argument('--output_dir', default='outputs/pretrain/', help='path where to save')
    parser.add_argument('--save_freq', default=1, type=int, help='save frequency')
    parser.add_argument('--step_size', default=3, type=int, help='step size for StepLR scheduler')
    parser.add_argument('--gamma', default=0.3, type=float, help='gamma for StepLR scheduler')
    parser.add_argument('--resume', help='resume from latest checkpoint', action='store_true')
    parser.add_argument("--sync_bn", help="use sync batch normalization", action="store_true")
    # dist
    parser.add_argument('--backend', default='nccl', type=str, help='backend for distributed training')
    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')
    return parser.parse_args()


def get_data_loader(args):
    
    if args.dataset == 'es_imagenet':  # downloaded
        train_dataset = es_imagenet.ESImageNet(root=args.root, train=True, nsteps=args.nsteps)
        val_dataset = es_imagenet.ESImageNet(root=args.root, train=False, nsteps=args.nsteps)
    else:
        raise NotImplementedError(args.dataset)
    
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
    else:
        train_sampler = torch.utils.data.RandomSampler(train_dataset)
        val_sampler = torch.utils.data.SequentialSampler(val_dataset)

    return DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler, num_workers=args.nworkers, pin_memory=True, drop_last=True), \
        DataLoader(val_dataset, batch_size=args.batch_size, sampler=val_sampler, num_workers=args.nworkers, pin_memory=True, drop_last=True), \

def _get_model(args):

    if args.model in sew_resnet.__dict__:
        model = sew_resnet.__dict__[args.model](num_classes=args.num_classes, T=args.nsteps, connect_f=args.connect_f)
    elif args.model in spiking_resnet.__dict__:
        model = spiking_resnet.__dict__[args.model](num_classes=args.num_classes, T=args.nsteps)
    else:
        raise NotImplementedError(args.model)

    return model


def _get_output_dir(args):

    output_dir = os.path.join(args.output_dir, f'{args.model}_{args.dataset}_lr{args.lr}_T{args.nsteps}')

    if args.model in sew_resnet.__dict__:
        output_dir += f'_cnf{args.connect_f}'
    
    return output_dir


def main(args):

    # init distributed training
    init_dist(args)
    print(args)
    
    # device
    if args.distributed:
        torch.cuda.set_device(args.local_rank)
    else:
        torch.cuda.set_device(args.device_id)

     # data
    train_loader, val_loader = get_data_loader(args)

    # criterion
    criterion = nn.CrossEntropyLoss()
    args.criterion = criterion.__class__.__name__
    
    # output dir
    output_dir = _get_output_dir(args)
    if is_master():
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print('Mkdir [{}]'.format(output_dir))
        if not os.path.exists(os.path.join(output_dir, 'checkpoints')):
            os.makedirs(os.path.join(output_dir, 'checkpoints'))
            print('Mkdir [{}]'.format(os.path.join(output_dir, 'checkpoints')))

    # model
    model = _get_model(args)
    
    # resume
    checkpoint = None
    if args.resume:
        checkpoint_files = glob.glob(os.path.join(output_dir, 'checkpoints/*.pth'))
        if checkpoint_files:
            latest_checkpoint = max(checkpoint_files, key=os.path.getctime)
            checkpoint = torch.load(latest_checkpoint, map_location='cpu')
            print('Resume from checkpoint [{}]'.format(latest_checkpoint))
            model.load_state_dict({k.replace('module.', ''):v for k, v in checkpoint['model'].items()})
            print('Resume model from epoch [{}]'.format(checkpoint['epoch']))

    model.cuda()
    if args.distributed and args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank)
    
    # run
    epoch = 0
    params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.Adam(params, lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    if checkpoint:
        epoch = checkpoint['epoch']
        optimizer.load_state_dict(checkpoint['optimizer'])
        print('Resume optimizer from epoch [{}]'.format(epoch))
        scheduler.load_state_dict(checkpoint['scheduler'])
        print('Resume scheduler from epoch [{}]'.format(epoch))


    # print and save args
    if is_master():
        with open(os.path.join(output_dir, 'config.yaml'), 'w') as f:
            yaml.dump(vars(args), f, default_flow_style=False)
        print('Args:' + str(vars(args)))
   

    train(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=train_loader,
        val_loader=val_loader,
        nepochs=args.nepochs,
        epoch=epoch,
        output_dir=output_dir,
        args=args
    )
    

if __name__ == '__main__':
    args = parser_args()
    main(args)