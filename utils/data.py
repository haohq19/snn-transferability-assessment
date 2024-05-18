import math
import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100, Caltech101, Caltech256, DTD, Food101
from spikingjelly.datasets import cifar10_dvs, dvs128_gesture, n_caltech101, n_mnist

# convert scaler to 1-hot vector
def to_onehot(y, nclasses):
    target = torch.zeros(y.size(0), nclasses)
    target[torch.arange(y.size(0)), y] = 1
    return target


def split2dataset(train_ratio: float, origin_dataset: torch.utils.data.Dataset, num_classes: int, random_split: bool = True):
    '''
    :param train_ratio: split the ratio of the origin dataset as the train set
    :type train_ratio: float
    :param origin_dataset: the origin dataset
    :type origin_dataset: torch.utils.data.Dataset
    :param num_classes: total classes number, e.g., ``10`` for the MNIST dataset
    :type num_classes: int
    :param random_split: If ``False``, the front ratio of samples in each classes will
            be included in train set, while the reset will be included in test set.
            If ``True``, this function will split samples in each classes randomly. The randomness is controlled by
            ``numpy.randon.seed``
    :type random_split: int
    :return: a tuple ``(train_set, test_set)``
    :rtype: tuple
    '''
    label_idx = []
    for i in range(num_classes):
        label_idx.append([])

    for i, item in enumerate(origin_dataset):
        y = item[1]
        if isinstance(y, np.ndarray) or isinstance(y, torch.Tensor):
            y = y.item()
        label_idx[y].append(i)
    train_idx = []
    test_idx = []
    if random_split:
        for i in range(num_classes):
            np.random.shuffle(label_idx[i])

    for i in range(num_classes):
        pos = math.ceil(label_idx[i].__len__() * train_ratio)
        train_idx.extend(label_idx[i][0: pos])
        test_idx.extend(label_idx[i][pos: label_idx[i].__len__()])

    return torch.utils.data.Subset(origin_dataset, train_idx), torch.utils.data.Subset(origin_dataset, test_idx)


def split3dataset(train_ratio: float, val_ratio: float, origin_dataset: torch.utils.data.Dataset, num_classes: int, random_split: bool = True):
    '''
    :param train_ratio: split the ratio of the origin dataset as the train set
    :param val_ratio: split the ratio of the origin dataset as the validation set
    :param origin_dataset: the origin dataset
    :param num_classes: total classes number
    :param random_split: If ``False``, the front ratio of samples in each classes will
            be included in train set, while the reset will be included in test set.
            If ``True``, this function will split samples in each classes randomly. The randomness is controlled by
            ``numpy.randon.seed``
    :return: a tuple ``(train_set, val_set, test_set)``
    '''
    label_idx = []
    for i in range(num_classes):
        label_idx.append([])

    for i, item in enumerate(origin_dataset): 
        y = item[1]  # item[1] is the label
        if isinstance(y, np.ndarray) or isinstance(y, torch.Tensor):
            y = y.item()  # convert to int
        label_idx[y].append(i)
    train_idx = []
    val_idx = []
    test_idx = []
    if random_split:
        for i in range(num_classes):
            np.random.shuffle(label_idx[i])

    for i in range(num_classes):
        pos_train = math.ceil(label_idx[i].__len__() * train_ratio)
        pos_val = math.ceil(label_idx[i].__len__() * (train_ratio + val_ratio))
        train_idx.extend(label_idx[i][0: pos_train])
        val_idx.extend(label_idx[i][pos_train: pos_val])
        test_idx.extend(label_idx[i][pos_val: label_idx[i].__len__()])

    return torch.utils.data.Subset(origin_dataset, train_idx), torch.utils.data.Subset(origin_dataset, val_idx), torch.utils.data.Subset(origin_dataset, test_idx)


def get_event_data_loader(args):

    if args.dataset == 'cifar10_dvs':  
        # downloaded
        # root = '/home/haohq/datasets/CIFAR10DVS'
        dataset = cifar10_dvs.CIFAR10DVS(root='/home/haohq/datasets/CIFAR10DVS', data_type='frame', frames_number=args.nsteps, split_by='time')
        train_dataset, val_dataset, test_dataset = split3dataset(0.8, 0.1, dataset, args.num_classes, random_split=False)

    elif args.dataset == 'dvs128_gesture':  
        # downloaded
        # root = '/home/haohq/datasets/DVS128Gesture'
        dataset = dvs128_gesture.DVS128Gesture
        train_dataset = dataset(root='/home/haohq/datasets/DVS128Gesture', train=True, data_type='frame', frames_number=args.nsteps, split_by='time')
        test_dataset = dataset(root='/home/haohq/datasets/DVS128Gesture', train=False, data_type='frame', frames_number=args.nsteps, split_by='time')
        train_dataset, val_dataset = split2dataset(0.9, train_dataset, args.num_classes, random_split=False)

    elif args.dataset == 'n_caltech101':  
        # downloaded
        # root = '/home/haohq/datasets/NCaltech101'
        dataset = n_caltech101.NCaltech101(root='/home/haohq/datasets/NCaltech101', data_type='frame', frames_number=args.nsteps, split_by='time')
        train_dataset, val_dataset, test_dataset= split3dataset(0.8, 0.1, dataset, args.num_classes, random_split=False)

    elif args.dataset == 'n_mnist':  
        # downloaded
        # root = '/home/haohq/datasets/NMNIST'
        train_dataset = n_mnist.NMNIST(root='/home/haohq/datasets/NMNIST', train=True, data_type='frame', frames_number=args.nsteps, split_by='time')
        test_dataset = n_mnist.NMNIST(root='/home/haohq/datasets/NMNIST', train=False, data_type='frame', frames_number=args.nsteps, split_by='time')
        train_dataset, val_dataset = split2dataset(0.9, train_dataset, args.num_classes, random_split=False)

    else:
        raise NotImplementedError(args.dataset)
    
    train_sampler = torch.utils.data.RandomSampler(train_dataset)
    val_sampler = torch.utils.data.SequentialSampler(val_dataset)
    test_sampler = torch.utils.data.SequentialSampler(test_dataset)

    return DataLoader(train_dataset, batch_size=128, sampler=train_sampler, num_workers=args.nworkers, pin_memory=True, drop_last=False), \
        DataLoader(val_dataset, batch_size=128, sampler=val_sampler, num_workers=args.nworkers, pin_memory=True, drop_last=False), \
        DataLoader(test_dataset, batch_size=128, sampler=test_sampler, num_workers=args.nworkers, pin_memory=True, drop_last=False)


def get_static_data_loader(args):

    _transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]
    )
])
    
    if args.dataset == 'cifar10':   # downloaded
        train_dataset = CIFAR10(train=True, root='/home/haohq/datasets/CIFAR10', download=True, transform=_transform)
        test_dataset = CIFAR10(train=False, root='/home/haohq/datasets/CIFAR10', download=True, transform=_transform)
        train_dataset, val_dataset = split2dataset(0.9, train_dataset, num_classes=args.num_classes, random_split=False)
    elif args.dataset == 'cifar100':  # downloaded
        train_dataset = CIFAR100(train=True, root='/home/haohq/datasets/CIFAR100', download=True, transform=_transform)
        test_dataset = CIFAR100(train=False, root='/home/haohq/datasets/CIFAR100', download=True, transform=_transform)
        train_dataset, val_dataset = split2dataset(0.9, train_dataset, num_classes=args.num_classes, random_split=False)
    elif args.dataset == 'dtd':  # downloaded
        train_dataset = DTD(split='train', root='/home/haohq/datasets/DTD', download=True, transform=_transform)
        val_dataset = DTD(split='val', root='/home/haohq/datasets/DTD', download=True, transform=_transform)
        test_dataset = DTD(split='test', root='/home/haohq/datasets/DTD', download=True, transform=_transform)
    elif args.dataset == 'food101':  # downloaded
        train_dataset = Food101(split='train', root='/home/haohq/datasets/Food101', download=True, transform=_transform)
        test_dataset = Food101(split='test', root='/home/haohq/datasets/Food101', download=True, transform=_transform)
        train_dataset, val_dataset = split2dataset(0.9, train_dataset, num_classes=args.num_classes, random_split=False)
    else:
        raise NotImplementedError(args.dataset)
    
    train_sampler = torch.utils.data.RandomSampler(train_dataset)
    val_sampler = torch.utils.data.SequentialSampler(val_dataset)
    test_sampler = torch.utils.data.SequentialSampler(test_dataset)

    return DataLoader(train_dataset, batch_size=128, sampler=train_sampler, num_workers=args.nworkers, pin_memory=True, drop_last=False), \
        DataLoader(val_dataset, batch_size=128, sampler=val_sampler, num_workers=args.nworkers, pin_memory=True, drop_last=False), \
        DataLoader(test_dataset, batch_size=128, sampler=test_sampler, num_workers=args.nworkers, pin_memory=True, drop_last=False)
