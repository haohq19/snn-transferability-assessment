# transfer pretrained SNN on static datasets

import os
import argparse
import numpy as np
import random
import hashlib
import torch
import torch.nn as nn
import models.spiking_resnet_static as spiking_resnet
import models.sew_resnet_static as sew_resnet
import utils
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from spikingjelly.activation_based import functional
from spikingjelly.activation_based import neuron, layer
from torchvision.datasets import CIFAR10, CIFAR100, Caltech101, Caltech256, DTD, Food101
from torch.utils.data import TensorDataset
from utils import split2dataset, split3dataset

_seed_ = 2020
random.seed(2020)
torch.manual_seed(_seed_)
torch.cuda.manual_seed_all(_seed_)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(_seed_)

from torchvision import transforms
_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]
    )
])


def parser_args():
    parser = argparse.ArgumentParser(description='transfer SNN on static datasets')
    # data
    parser.add_argument('--dataset', default='food101', type=str, help='dataset')
    parser.add_argument('--nsteps', default=4, type=int, help='number of time steps')
    parser.add_argument('--nclasses', default=101, type=int, help='number of classes')
    parser.add_argument('--batch_size', default=1024, type=int, help='batch size')
    # model
    parser.add_argument('--model', default='sew_resnet18', type=str, help='model type')
    parser.add_argument('--pretrained_path', default='/home/haohq/SNN-Trans-Assess/weights/sew18_checkpoint_319.pth', type=str, help='path to pre-trained weights')
    parser.add_argument('--connect_f', default='ADD', type=str, help='spike-element-wise connect function')
    parser.add_argument('--num_dims', default=512, type=int, help='number of dimensions')
    
    # run
    parser.add_argument('--device_id', default=0, type=int, help='GPU id to use.')
    parser.add_argument('--nepochs', default=100, type=int, help='number of epochs')
    parser.add_argument('--nworkers', default=16, type=int, help='number of workers')
    parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
    parser.add_argument('--weight_decay', default=5e-6, type=float, help='weight decay')
    parser.add_argument('--optim', default='Adam', type=str, help='optimizer')
    parser.add_argument('--output_dir', default='output', help='path where to save')
    return parser.parse_args()


class LinearProbe(nn.Module):
    def __init__(self, in_features, num_classes):
        super().__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        self.fc = layer.SeqToANNContainer(
            nn.Linear(in_features, num_classes)
            )
        self.sn = neuron.IFNode(step_mode='m', detach_reset=True)

    def forward(self, features):
        # feature.shape: nsteps, batch_size, in_features
        x = self.fc(features)
        x = self.sn(x)
        return x.mean(dim=0)  # batch_size, num_classes


def load_data(args):
    
    if args.dataset == 'cifar10':   # downloaded
        train_dataset = CIFAR10(train=True, root='/home/haohq/datasets/CIFAR10', download=True, transform=_transform)
        test_dataset = CIFAR10(train=False, root='/home/haohq/datasets/CIFAR10', download=True, transform=_transform)
        train_dataset, val_dataset = split2dataset(0.9, train_dataset, num_classes=args.nclasses, random_split=False)
    elif args.dataset == 'cifar100':  # downloaded
        train_dataset = CIFAR100(train=True, root='/home/haohq/datasets/CIFAR100', download=True, transform=_transform)
        test_dataset = CIFAR100(train=False, root='/home/haohq/datasets/CIFAR100', download=True, transform=_transform)
        train_dataset, val_dataset = split2dataset(0.9, train_dataset, num_classes=args.nclasses, random_split=False)
    elif args.dataset == 'dtd':  # downloaded
        train_dataset = DTD(split='train', root='/home/haohq/datasets/DTD', download=True, transform=_transform)
        val_dataset = DTD(split='val', root='/home/haohq/datasets/DTD', download=True, transform=_transform)
        test_dataset = DTD(split='test', root='/home/haohq/datasets/DTD', download=True, transform=_transform)
    elif args.dataset == 'food101':  # downloaded
        train_dataset = Food101(split='train', root='/home/haohq/datasets/Food101', download=True, transform=_transform)
        test_dataset = Food101(split='test', root='/home/haohq/datasets/Food101', download=True, transform=_transform)
        train_dataset, val_dataset = split2dataset(0.9, train_dataset, num_classes=args.nclasses, random_split=False)
    else:
        raise NotImplementedError(args.dataset)
    
    train_sampler = torch.utils.data.RandomSampler(train_dataset)
    val_sampler = torch.utils.data.SequentialSampler(val_dataset)
    test_sampler = torch.utils.data.SequentialSampler(test_dataset)

    return DataLoader(train_dataset, batch_size=128, sampler=train_sampler, num_workers=args.nworkers, pin_memory=True, drop_last=False), \
        DataLoader(val_dataset, batch_size=128, sampler=val_sampler, num_workers=args.nworkers, pin_memory=True, drop_last=False), \
        DataLoader(test_dataset, batch_size=128, sampler=test_sampler, num_workers=args.nworkers, pin_memory=True, drop_last=False)

def load_model(args):
    # load pretrained weights
    if not os.path.exists(args.pretrained_path):
        raise FileNotFoundError(args.pretrained_path)
    else:
        checkpoint = torch.load(args.pretrained_path)
        if 'model' in checkpoint.keys():
            pretrained_weights = checkpoint['model']
        else:
            pretrained_weights = checkpoint

    # model
    if args.model in sew_resnet.__dict__:
        model = sew_resnet.__dict__[args.model](num_classes=1000, T=args.nsteps, connect_f=args.connect_f)
        if pretrained_weights is not None:
            state_dict = {k.replace('module.', ''): v for k, v in pretrained_weights.items()}
            state_dict = {k.replace('fc', 'fc.0'): v for k, v in state_dict.items()}
            model.load_state_dict(state_dict)
            print('load pretrained weights from {}'.format(args.pretrained_path))
    elif args.model in spiking_resnet.__dict__:
        model = spiking_resnet.__dict__[args.model](num_classes=1000, T=args.nsteps)
        if pretrained_weights is not None:
            state_dict = {k.replace('module.', ''): v for k, v in pretrained_weights.items()}
            state_dict = {k.replace('fc', 'fc.0'): v for k, v in state_dict.items()}
            model.load_state_dict(state_dict)
            print('load pretrained weights from {}'.format(args.pretrained_path))
    else:
        raise NotImplementedError(args.model)
    
    for name, param in model.named_parameters():
        param.requires_grad = False

    return model


def _get_output_dir(args):

    output_dir = os.path.join(args.output_dir, args.dataset)
    output_dir = os.path.join(output_dir, args.model)
    output_dir = os.path.join(output_dir, f'lr{args.lr}_wd{args.weight_decay}_ep{args.nepochs}_T{args.nsteps}')

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

    # pretrained
    
    sha256_hash = hashlib.sha256(args.pretrained_path.encode()).hexdigest()
    output_dir += '_pt'
    output_dir += f'_{sha256_hash[:8]}'
    
    return output_dir


def cache(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    cache_dir: str,
):  
    if os.path.exists(os.path.join(cache_dir, 'cache')):
        print('cached feature map already exists')
        train_dataset = TensorDataset(torch.from_numpy(np.load(os.path.join(cache_dir, 'cache/train_features.npy'))), torch.from_numpy(np.load(os.path.join(cache_dir, 'cache/train_labels.npy'))))
        val_dataset = TensorDataset(torch.from_numpy(np.load(os.path.join(cache_dir, 'cache/val_features.npy'))), torch.from_numpy(np.load(os.path.join(cache_dir, 'cache/val_labels.npy'))))
        test_dataset = TensorDataset(torch.from_numpy(np.load(os.path.join(cache_dir, 'cache/test_features.npy'))), torch.from_numpy(np.load(os.path.join(cache_dir, 'cache/test_labels.npy'))))
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.nworkers, pin_memory=True, drop_last=False)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.nworkers, pin_memory=True, drop_last=False)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.nworkers, pin_memory=True, drop_last=False)
        return train_loader, val_loader, test_loader
    else:
        os.makedirs(os.path.join(cache_dir, 'cache'))
    
    torch.cuda.empty_cache()
    with torch.no_grad():
        model.eval()

        # train_loader
        features = []
        logits = []
        labels = []
        nsteps_per_epoch = len(train_loader)
        if utils.is_master():
            import tqdm
            process_bar = tqdm.tqdm(total=nsteps_per_epoch)
        for input, label in train_loader:
            input = input.cuda(non_blocking=True)
            label = label.numpy()
            output = model(input)  # N, C
            feature_map = model.feature.detach().cpu().numpy()  # T, N, D
            features.append(feature_map)
            logit = output.softmax(dim=1).detach().cpu().numpy()  # N, C
            logits.append(logit)
            labels.append(label)
            functional.reset_net(model)

            process_bar.update(1)
        process_bar.close()
        
        features = np.concatenate(features, axis=1)  # T, N, D
        features = features.transpose(1, 0, 2)  # N, T, D
        logits = np.concatenate(logits, axis=0) # N, num_classes
        labels = np.concatenate(labels, axis=0)  # N
        np.save(os.path.join(cache_dir, 'cache/train_features.npy'), features)
        np.save(os.path.join(cache_dir, 'cache/train_logits.npy'), logits)
        np.save(os.path.join(cache_dir, 'cache/train_labels.npy'), labels)

        # val_loader
        features = []
        labels = []
        nsteps_per_epoch = len(val_loader)
        if utils.is_master():
            import tqdm
            process_bar = tqdm.tqdm(total=nsteps_per_epoch)
        for input, label in val_loader:
            input = input.cuda(non_blocking=True)
            label = label.numpy()
            model(input)
            feature_map = model.feature.detach().cpu().numpy()
            features.append(feature_map)
            labels.append(label)

            functional.reset_net(model)
            process_bar.update(1)
        process_bar.close()
        features = np.concatenate(features, axis=1)
        features = features.transpose(1, 0, 2)
        labels = np.concatenate(labels, axis=0)
        np.save(os.path.join(cache_dir, 'cache/val_features.npy'), features)
        np.save(os.path.join(cache_dir, 'cache/val_labels.npy'), labels)

        # test_loader
        features = []
        labels = []
        nsteps_per_epoch = len(test_loader)
        if utils.is_master():
            import tqdm
            process_bar = tqdm.tqdm(total=nsteps_per_epoch)
        for input, label in test_loader:
            input = input.cuda(non_blocking=True)
            label = label.numpy()
            model(input)
            feature_map = model.feature.detach().cpu().numpy()
            features.append(feature_map)
            labels.append(label)

            functional.reset_net(model)
            process_bar.update(1)
        process_bar.close()
        features = np.concatenate(features, axis=1)
        features = features.transpose(1, 0, 2)
        labels = np.concatenate(labels, axis=0)
        np.save(os.path.join(cache_dir, 'cache/test_features.npy'), features)
        np.save(os.path.join(cache_dir, 'cache/test_labels.npy'), labels)

        print('cached feature map saved to {}'.format(os.path.join(cache_dir, 'cache')))
    
    train_dataset = TensorDataset(torch.from_numpy(np.load(os.path.join(cache_dir, 'cache/train_features.npy'))), torch.from_numpy(np.load(os.path.join(cache_dir, 'cache/train_labels.npy'))))
    val_dataset = TensorDataset(torch.from_numpy(np.load(os.path.join(cache_dir, 'cache/val_features.npy'))), torch.from_numpy(np.load(os.path.join(cache_dir, 'cache/val_labels.npy'))))
    test_dataset = TensorDataset(torch.from_numpy(np.load(os.path.join(cache_dir, 'cache/test_features.npy'))), torch.from_numpy(np.load(os.path.join(cache_dir, 'cache/test_labels.npy'))))
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.nworkers, pin_memory=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.nworkers, pin_memory=True, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.nworkers, pin_memory=True, drop_last=False)
    return train_loader, val_loader, test_loader

def train(
    model: nn.Module,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_loader: DataLoader,
    val_loader: DataLoader,
    nepochs: int,
    epoch: int,
    output_dir: str,
    args: argparse.Namespace,
):  
    
    tb_writer = SummaryWriter(output_dir + '/log')
    print('log saved to {}'.format(output_dir + '/log'))
    # save the pretrained_path to output_dir
    with open(os.path.join(output_dir, 'pretrained_path.txt'), 'w') as f:
        f.write(args.pretrained_path)

    torch.cuda.empty_cache()
    # train 
    epoch = epoch
    best_model = None
    best_acc = 0
    while(epoch < nepochs):
        print('Epoch {}/{}'.format(epoch+1, nepochs))
        model.train()
        top1_correct = 0
        top5_correct = 0
        total = len(train_loader.dataset)
        total_loss = 0
        nsteps_per_epoch = len(train_loader)
        step = 0
        import tqdm
        process_bar = tqdm.tqdm(total=nsteps_per_epoch)
        for input, label in train_loader:
            input = input.cuda(non_blocking=True)
            label = label.cuda(non_blocking=True)
            target = utils.to_onehot(label, args.nclasses).cuda(non_blocking=True)
            input = input.transpose(0, 1)  # T, N, D
            output = model(input)
            loss = criterion(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            functional.reset_net(model)

            # calculate the top5 and top1 accurate numbers
            _, predicted = output.topk(5, 1, True, True)
            top1_correct += predicted[:, 0].eq(label).sum().item()
            top5_correct += predicted.T.eq(label[None]).sum().item()
            total_loss += loss.item() * input.shape[1]
            step += 1
            tb_writer.add_scalar('step_loss', loss.item(), epoch * nsteps_per_epoch + step)
            process_bar.update(1)
        top1_accuracy = top1_correct / total
        top5_accuracy = top5_correct / total
        total_loss = total_loss / total 
        tb_writer.add_scalar('train_acc@1', top1_accuracy, epoch + 1)
        tb_writer.add_scalar('train_acc@5', top5_accuracy, epoch + 1)
        tb_writer.add_scalar('train_loss', total_loss, epoch + 1)
        process_bar.close()
        print('train_cor@1: {}, train_cor@5: {}, train_total: {}'.format(top1_correct, top5_correct, total))
        print('train_acc@1: {}, train_acc@5: {}, train_loss: {}'.format(top1_accuracy, top5_accuracy, total_loss))
        
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
                target = utils.to_onehot(label, args.nclasses).cuda(non_blocking=True)
                input = input.transpose(0, 1)  # T, N, D
                output = model(input)
                loss = criterion(output, target)
                functional.reset_net(model)

                # calculate the top5 and top1 accurate numbers
                _, predicted = output.topk(5, 1, True, True)  # batch_size, topk(5) 
                top1_correct += predicted[:, 0].eq(label).sum().item()
                top5_correct += predicted.T.eq(label[None]).sum().item()
                total_loss += loss.item() * input.shape[1]
        top1_accuracy = top1_correct / total
        top5_accuracy = top5_correct / total
        total_loss = total_loss / total
        tb_writer.add_scalar('val_acc@1', top1_accuracy, epoch + 1)
        tb_writer.add_scalar('val_acc@5', top5_accuracy, epoch + 1)
        tb_writer.add_scalar('val_loss', total_loss, epoch + 1)
        print('val_cor@1: {}, val_cor@5: {}, val_total: {}'.format(top1_correct, top5_correct, total))
        print('val_acc@1: {}, val_acc@5: {}, val_loss: {}'.format(top1_accuracy, top5_accuracy, total_loss))

        epoch += 1

        # save best
        if top1_accuracy >= best_acc:
            best_acc = top1_accuracy
            best_model = model
            checkpoint = {
                'model': best_model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'args': args,
            }
            save_name = 'checkpoint/best_{}.pth'.format(top1_accuracy)
            torch.save(checkpoint, os.path.join(output_dir, save_name))
            print('saved best model to {}'.format(output_dir))
    print('best_val_acc@1: {}'.format(best_acc))
    return best_model


def test(
    model: nn.Module,
    test_loader: DataLoader,
    output_dir: str,
):
    top1_correct = 0
    top5_correct = 0
    total = len(test_loader.dataset)
    with torch.no_grad():
        for input, label in test_loader:
            input = input.cuda(non_blocking=True)
            label = label.cuda(non_blocking=True)
            input = input.transpose(0, 1)  # T, N, D
            output = model(input)
            functional.reset_net(model)

            # calculate the top5 and top1 accurate numbers
            _, predicted = output.topk(5, 1, True, True)
            top1_correct += predicted[:, 0].eq(label).sum().item()
            top5_correct += predicted.T.eq(label[None]).sum().item()

    top1_accuracy = top1_correct / total
    top5_accuracy = top5_correct / total
    print('test_cor@1: {}, test_cor@5: {}, test_total: {}'.format(top1_correct, top5_correct, total))
    print('test_acc@1: {}, test_acc@5: {}'.format(top1_accuracy, top5_accuracy))
    with open(os.path.join(output_dir, 'test_acc.txt'), 'w') as f:
        f.write('test_cor@1: {}, test_cor@5: {}, test_total: {}'.format(top1_correct, top5_correct, total))
        f.write('\n')
        f.write('test_acc@1: {}, test_acc@5: {}'.format(top1_accuracy, top5_accuracy))
            



def main(args):
    print(args)
    torch.cuda.set_device(args.device_id)

    # criterion
    criterion = nn.MSELoss()
    args.criterion = criterion.__class__.__name__
    
    # output
    output_dir = _get_output_dir(args)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(os.path.join(output_dir, 'checkpoint')):
        os.makedirs(os.path.join(output_dir, 'checkpoint'))

    # model
    model = load_model(args)
    model.cuda()
    
     # data
    cache_dir = os.path.join(args.output_dir, args.dataset)
    cache_dir = os.path.join(cache_dir, args.model)
    train_loader, val_loader, test_loader = None, None, None
    if not os.path.exists(os.path.join(cache_dir, 'cache')):
        train_loader, val_loader, test_loader = load_data(args)
    train_loader, val_loader, test_loader = cache(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        cache_dir=cache_dir,
    )

    # linear probe
    linear_probe = LinearProbe(args.num_dims, args.nclasses)
    linear_probe.cuda()
    
    # run
    epoch = 0
    optim = args.optim
    sched = args.sched
    params = filter(lambda p: p.requires_grad, linear_probe.parameters())
    if optim == 'SGD':
        optimizer = torch.optim.SGD(params, lr=args.lr,  weight_decay=args.weight_decay)
    elif optim == 'Adam':
        optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
    else:
        raise NotImplementedError(optim)
    
    if sched == 'StepLR':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    else:
        raise NotImplementedError(sched)
   

    best_model = train(
        model=linear_probe,
        criterion=criterion,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        nepochs=args.nepochs,
        epoch=epoch,
        output_dir=output_dir,
        args=args,
    )
    test(
        model=best_model,
        test_loader=test_loader,
        output_dir=output_dir,
        args=args,
    )
    

if __name__ == '__main__':
    args = parser_args()
    main(args)


'''
python transfer_static.py 
'''