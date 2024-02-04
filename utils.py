import os
import math
import torch
import numpy as np
import torch.distributed as dist

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


def is_master():
    if not dist.is_available():  # if not distributed mode, return True
        return True
    elif not dist.is_initialized():  # if distributed mode but not initialized, return True
        return True
    else:  # if distributed mode, return True only when rank is 0
        return dist.get_rank() == 0

def save_on_master(*args, **kwargs):
    if is_master():
        torch.save(*args, **kwargs)
        
def init_dist(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:  # distributed mode
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.local_rank = int(os.environ['LOCAL_RANK'])
        args.distributed = True
    else:  # not distributed mode
        args.distributed = False
        return

    torch.cuda.set_device(args.local_rank)
    print('| distributed init (rank {}, local rank {}): {}'.format(args.rank, args.local_rank, args.dist_url), flush=True)
    dist.init_process_group(backend=args.backend, init_method=args.dist_url, world_size=args.world_size, rank=args.rank)
    enable_print(is_master())
    
    
def enable_print(is_master):
    '''
    This function disables printing when not in master process
    '''
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def global_meters_all_avg(args, *meters):
    '''
    meters: scalar values of loss/accuracy calculated in each rank
    '''
    tensors = [torch.tensor(meter, device=args.local_rank, dtype=torch.float32) for meter in meters]
    for tensor in tensors:
        # each item of `tensors` is all-reduced starting from index 0 (in-place)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

    return [(tensor / args.world_size).item() for tensor in tensors]


def global_meters_all_sum(args, *meters):
    '''
    meters: scalar values calculated in each rank
    '''
    tensors = [torch.tensor(meter, device=args.local_rank, dtype=torch.float32) for meter in meters]
    for tensor in tensors:
        # each item of `tensors` is all-reduced starting from index 0 (in-place)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    
    return [tensor.item() for tensor in tensors]