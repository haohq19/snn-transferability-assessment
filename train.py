import os
import argparse
import numpy as np
import random
import glob
import torch
import torch.nn as nn
import spiking_resnet, sew_resnet
import es_imagenet
import utils
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from spikingjelly.activation_based import functional
from spikingjelly.datasets import asl_dvs, cifar10_dvs, dvs128_gesture, n_caltech101, n_mnist
from utils import split2dataset, split3dataset

_seed_ = 2020
random.seed(2020)
torch.manual_seed(_seed_)
torch.cuda.manual_seed_all(_seed_)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(_seed_)

def parser_args():
    parser = argparse.ArgumentParser(description='Fine-tune a SNN')
    # data
    parser.add_argument('--dataset', default='n_caltech101', type=str, help='dataset')
    parser.add_argument('--root', default='/home/haohq/datasets/NCaltech101', type=str, help='path to dataset')
    parser.add_argument('--duration', default=50, type=int, help='duration of a single frame (ms)')
    parser.add_argument('--nsteps', default=8, type=int, help='number of time steps')
    parser.add_argument('--nclasses', default=101, type=int, help='number of classes')
    parser.add_argument('--batch_size', default=8, type=int, help='batch size')
    # model
    parser.add_argument('--model', default='sew_resnet101', type=str, help='model type (default: sew_resnet18)')
    parser.add_argument('--save_path', default='weights/sew50_checkpoint_319.pth', type=str, help='path to saved weights')
    parser.add_argument('--connect_f', default='ADD', type=str, help='spike-element-wise connect function')
    # run
    parser.add_argument('--device_id', default=6, type=int, help='GPU id to use.')
    parser.add_argument('--nepochs', default=100, type=int, help='number of epochs')
    parser.add_argument('--nworkers', default=16, type=int, help='number of workers')
    parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--weight_decay', default=0, type=float, help='weight decay')
    parser.add_argument('--optim', default='Adam', type=str, help='optimizer')
    parser.add_argument('--output_dir', default='output/', help='path where to save')
    parser.add_argument('--save_freq', default=10, type=int, help='save frequency')
    parser.add_argument('--sched', default='StepLR', type=str, help='scheduler')
    parser.add_argument('--step_size', default=40, type=int, help='step size for scheduler')
    parser.add_argument('--gamma', default=0.1, type=float, help='gamma for scheduler')
    parser.add_argument('--resume', help='resume from checkpoint', action='store_true')
    # dist
    parser.add_argument('--world-size', default=10, type=int, help='number of distributed processes')
    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')
    parser.add_argument("--sync_bn", help="use sync batch normalization", action="store_true")
    return parser.parse_args()


def load_data(args):
    
    if args.dataset == 'asl_dvs':  
        dataset = asl_dvs.ASLDVS(root=args.root, data_type='frame', frames_number=args.nsteps, split_by='time', duration=args.duration)
        train_dataset, val_dataset, test_dataset = split3dataset(0.8, 0.1, dataset, args.nclasses, random_split=False)
    elif args.dataset == 'cifar10_dvs':  # downloaded
        dataset = cifar10_dvs.CIFAR10DVS(root=args.root, data_type='frame', frames_number=args.nsteps, split_by='time', duration=args.duration)
        train_dataset, val_dataset, test_dataset = split3dataset(0.8, 0.1, dataset, args.nclasses, random_split=False)
    elif args.dataset == 'dvs128_gesture':  # downloaded
        dataset = dvs128_gesture.DVS128Gesture
        train_dataset = dataset(root=args.root, train=True, data_type='frame', duration=args.duration)
        test_dataset = dataset(root=args.root, train=False, data_type='frame', duration=args.duration)
        train_dataset, val_dataset = split2dataset(0.9, train_dataset, args.nclasses, random_split=True)
    elif args.dataset == 'n_caltech101':  # downloaded
        dataset = n_caltech101.NCaltech101(root=args.root, data_type='frame', frames_number=args.nsteps, split_by='time', duration=args.duration)
        train_dataset, val_dataset, test_dataset= split3dataset(0.8, 0.1, dataset, args.nclasses, random_split=False)
    elif args.dataset == 'n_mnist':  # downloaded
        train_dataset = n_mnist.NMNIST(root=args.root, train=True, data_type='frame', frames_number=args.nsteps, split_by='time', duration=args.duration)
        test_dataset = n_mnist.NMNIST(root=args.root, train=False, data_type='frame', frames_number=args.nsteps, split_by='time', duration=args.duration)
        train_dataset, val_dataset = split2dataset(0.9, train_dataset, args.nclasses, random_split=True)
    elif args.dataset == 'es_imagenet':  # downloaded
        train_dataset = es_imagenet.ESImageNet(root=args.root, train=True, nsteps=args.nsteps)
        test_dataset = es_imagenet.ESImageNet(root=args.root, train=False, nsteps=args.nsteps)
        train_dataset, val_dataset = train_dataset.split(0.9, random_split=True)
    else:
        raise NotImplementedError(args.dataset)
    
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
        test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)
    else:
        train_sampler = torch.utils.data.RandomSampler(train_dataset)
        val_sampler = torch.utils.data.SequentialSampler(val_dataset)
        test_sampler = torch.utils.data.SequentialSampler(test_dataset)

    return DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler, num_workers=args.nworkers, pin_memory=True), \
        DataLoader(val_dataset, batch_size=args.batch_size, sampler=val_sampler, num_workers=args.nworkers, pin_memory=True), \
        DataLoader(test_dataset, batch_size=args.batch_size, sampler=test_sampler, num_workers=args.nworkers, pin_memory=True)

def load_model(args):
    # load weights
    save = torch.load(args.save_path)
    weight = save['model']

    # model
    if args.model in sew_resnet.__dict__:
        model = sew_resnet.__dict__[args.model](num_classes=args.nclasses, T=args.nsteps, connect_f=args.connect_f)
        # params = model.state_dict()
        # for k, v in weight.items():
        #     if k in params:
        #         params[k] = v
        # model.load_state_dict(params)
    elif args.model in spiking_resnet.__dict__:
        model = spiking_resnet.__dict__[args.model](num_classes=args.nclasses, T=args.nsteps)
        # raise NotImplementedError(args.model)
    else:
        raise NotImplementedError(args.model)
    
    # freeze the weights
    # for name, param in model.named_parameters():
    #     if 'layer' in name and 'layer1' not in name:
    #         param.requires_grad = False

    return model


def _get_output_dir(args):

    output_dir = os.path.join(args.output_dir, f'{args.model}_{args.dataset}_b{args.batch_size}_lr{args.lr}_T{args.nsteps}')
    
    if args.weight_decay:
        output_dir += f'_wd{args.weight_decay}'

    # criterion
    if args.criterion == 'CrossEntropyLoss':
        output_dir += '_CE'
    elif args.criterion == 'MSELoss':
        output_dir += '_MSE'
    else:
        raise NotImplementedError(args.criterion)

    # optimizer
    if args.optim == 'Adam':
        output_dir += '_adam'
    elif args.optim == 'SGD':
        output_dir += '_sgd'
    else:
        raise NotImplementedError(args.optim)

    if args.momentum:
        output_dir += f'_mom{args.momentum}'

    # scheduler
    if args.sched == 'StepLR':
        output_dir += f'_step{args.step_size}_gamma{args.gamma}'
    else:
        raise NotImplementedError(args.sched)

    if args.connect_f:
        output_dir += f'_cnf{args.connect_f}'
    
    return output_dir

def train(
    model: nn.Module,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    train_loader: DataLoader,
    val_loader: DataLoader,
    nepochs: int,
    epoch: int,
    # tb_writer: SummaryWriter,
    output_dir: str,
    args: argparse.Namespace,
):  
    if utils.is_master():
        tb_writer = SummaryWriter(output_dir + '/log')
        print('log saved to {}'.format(output_dir + '/log'))
        
        
    # train 
    epoch = epoch
    while(epoch < nepochs):
        print('Epoch {}/{}'.format(epoch+1, nepochs))
        model.train()
        top1_correct = 0
        top5_correct = 0
        total = len(train_loader.dataset)
        total_loss = 0
        nsteps_per_epoch = len(train_loader)
        step = 0
        if utils.is_master():
            import tqdm
            process_bar = tqdm.tqdm(total=nsteps_per_epoch)
        for input, label in train_loader:
            input = input.cuda(non_blocking=True)
            label = label.cuda(non_blocking=True)
            input = input.transpose(0, 1)
            target = utils.to_onehot(label, args.nclasses).cuda(non_blocking=True)
            # target = target.cuda(non_blocking=True)
            output = model(input).mean(dim=0).squeeze()
            loss = criterion(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            functional.reset_net(model)

            # calculate the top5 and top1 accurate numbers
            _, predicted = output.topk(5, 1, True, True)
            top1_correct += predicted[:, 0].eq(label).sum().item()
            top5_correct += predicted.T.eq(label[None]).sum().item()
            total_loss += loss.item()
            step += 1
            if utils.is_master():
                tb_writer.add_scalar('step_loss', loss.item(), epoch * nsteps_per_epoch + step)
                process_bar.update(1)
        if args.distributed:
            top1_correct, top5_correct, total_loss = utils.global_meters_all_sum(args, top1_correct, top5_correct, total_loss)
        top1_accuracy = top1_correct / total * 100
        top5_accuracy = top5_correct / total * 100
        if utils.is_master():    
            tb_writer.add_scalar('train_acc@1', top1_accuracy, epoch + 1)
            tb_writer.add_scalar('train_acc@5', top5_accuracy, epoch + 1)
            tb_writer.add_scalar('train_loss', total_loss, epoch + 1)
            process_bar.close()
        print('train_cor@1: {}, train_cor@5: {}, train_total: {}'.format(top1_correct, top5_correct, total))
        print('train_acc@1: {:.3f}%, train_acc@5: {:.3f}%, train_loss: {:.3f}'.format(top1_accuracy, top5_accuracy, total_loss))
        
        # evaluate
        model.eval()
        top1_correct = 0
        top5_correct = 0
        total = len(val_loader.dataset)
        total_loss = 0
        with torch.no_grad():
            for input, label in val_loader:
                input = input.cuda(non_blocking=True)
                label = label.cuda(non_blocking=True)
                input = input.transpose(0, 1)
                target = utils.to_onehot(label, args.nclasses).cuda(non_blocking=True)
                output = model(input).mean(dim=0).squeeze()  # batch_size, num_classes
                loss = criterion(output, target)
                functional.reset_net(model)

                # calculate the top5 and top1 accurate numbers
                _, predicted = output.topk(5, 1, True, True)  # batch_size, topk(5) 
                top1_correct += predicted[:, 0].eq(label).sum().item()
                top5_correct += predicted.T.eq(label[None]).sum().item()
                total_loss += loss.item()
        if args.distributed:
            top1_correct, top5_correct, total_loss = utils.global_meters_all_sum(args, top1_correct, top5_correct, total_loss)
        top1_accuracy = top1_correct / total * 100
        top5_accuracy = top5_correct / total * 100
        if utils.is_master():   
            tb_writer.add_scalar('val_acc@1', top1_accuracy, epoch + 1)
            tb_writer.add_scalar('val_acc@5', top5_accuracy, epoch + 1)
            tb_writer.add_scalar('val_loss', total_loss, epoch + 1)
        print('val_cor@1: {}, val_cor@5: {}, val_total: {}'.format(top1_correct, top5_correct, total))
        print('val_acc@1: {:.3f}%, val_acc@5: {:.3f}%, val_loss: {:.3f}'.format(top1_accuracy, top5_accuracy, total_loss))

        
        # save
        epoch += 1
        scheduler.step()
        if epoch % args.save_freq == 0:
            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'epoch': epoch,
                'args': args,
            }
            save_name = 'checkpoint/checkpoint_epoch{}_valacc{:.2f}.pth'.format(epoch, top1_accuracy)
            utils.save_on_master(checkpoint, os.path.join(output_dir, save_name))
            print('saved checkpoint to {}'.format(output_dir))


def test(
    model: nn.Module,
    test_loader: DataLoader,
    args: argparse.Namespace,
):
    top1_correct = 0
    top5_correct = 0
    total = len(test_loader.dataset)
    with torch.no_grad():
        for input, label in test_loader:
            input = input.cuda(non_blocking=True)
            label = label.cuda(non_blocking=True)
            input = input.transpose(0, 1)
            output = model(input).mean(dim=0).squeeze()
            functional.reset_net(model)

            # calculate the top5 and top1 accurate numbers
            _, predicted = output.topk(5, 1, True, True)
            top1_correct += predicted[:, 0].eq(label).sum().item()
            top5_correct += predicted.T.eq(label[None]).sum().item()

    if args.distributed:
        top1_correct, top5_correct = utils.global_meters_all_sum(args, top1_correct, top5_correct)
    top1_accuracy = top1_correct / total * 100
    top5_accuracy = top5_correct / total * 100
    print('test_cor@1: {}, test_cor@5: {}, test_total: {}'.format(top1_correct, top5_correct, total))
    print('test_acc@1: {:.3f}%, test_acc@5: {:.3f}%'.format(top1_accuracy, top5_accuracy))


def main(args):
    # init distributed training
    utils.init_dist(args)
    print(args)
    
    # device
    if args.distributed:
        torch.cuda.set_device(args.local_rank)
    else:
        torch.cuda.set_device(args.device_id)

     # data
    train_loader, val_loader, test_loader = load_data(args)

    # model
    model = load_model(args)
    model.cuda()
    if args.distributed and args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank])
    # run
    optim = args.optim
    sched = args.sched
    criterion = nn.CrossEntropyLoss()
    args.criterion = criterion.__class__.__name__
    params = filter(lambda p: p.requires_grad, model.parameters())
    if optim == 'SGD':
        optimizer = torch.optim.SGD(params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    elif optim == 'Adam':
        optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
    else:
        raise NotImplementedError(optim)
    if sched == 'StepLR':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    else:
        raise NotImplementedError(sched)


    # save
    output_dir = _get_output_dir(args)
    if utils.is_master():
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        if not os.path.exists(os.path.join(output_dir, 'checkpoint')):
            os.makedirs(os.path.join(output_dir, 'checkpoint'))
    
    # resume
    epoch = 0
    if args.resume:
        checkpoints = glob.glob(os.path.join(output_dir, 'checkpoint/*.pth'))
        if checkpoints:
            latest_checkpoint = max(checkpoints, key=os.path.getctime)
            state_dict = torch.load(latest_checkpoint)
            if args.distributed:
                model.load_state_dict(state_dict['model'])
            else:
                model.load_state_dict({k.replace('module.', ''):v for k, v in state_dict['model'].items()})
            optimizer.load_state_dict(state_dict['optimizer'])
            scheduler.load_state_dict(state_dict['scheduler'])
            epoch = state_dict['epoch']
            print('load checkpoint from {}'.format(latest_checkpoint))

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
    test(
        model=model,
        test_loader=test_loader,
        args=args
    )
    

if __name__ == '__main__':
    args = parser_args()
    main(args)


'''
python -m torch.distributed.run --nproc_per_node=8 train.py --sync_bn --batch_size 40
'''