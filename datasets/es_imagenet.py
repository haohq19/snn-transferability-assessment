import linecache
import numpy as np
import torch
import os
import math


class ESImageNet(torch.utils.data.Dataset):
    def __init__(
            self,
            root: str,
            train: bool = True,
            nsteps: int = 8,
            ):
        '''
        param root: the root path of the dataset
        param train: if True, return the training set, else return the validation set, default True
        param nsteps: the number of steps of the input data, default 8
        '''
        super(ESImageNet).__init__()

        self.root = root
        self.train = train
        self.train_label_path = os.path.join(self.root, 'trainlabel.txt')
        self.val_label_path = os.path.join(self.root, 'vallabel.txt')
        self.nsteps = nsteps
        if self.nsteps != 8:
            raise NotImplementedError('nsteps must be 8')
        file_names = []
        if self.train:
            self.path = os.path.join(self.root, 'train')
            with open(self.train_label_path, 'r') as file:
                for line in file:
                    file_name, _, _, _ = line.split()  # file_name, class_num, width, height
                    file_names.append(file_name)
        else:
            self.path = os.path.join(self.root, 'val')
            with open(self.val_label_path, 'r') as file:
                for line in file:
                    file_name, _, _, _ = line.split()
                    file_names.append(file_name)
        self.__len__ = len(file_names)

    
        
        
    def __getitem__(self, idx):
        '''
        return input data of size (nsteps, in_channels=2, width=224, height=224) and label
        '''
        if self.train:
            line = linecache.getline(self.train_label_path, idx + 1)
        else:
            line = linecache.getline(self.val_label_path, idx + 1)
        file_name, class_num, width, height = line.split()
        file_path = os.path.join(self.path, file_name)
        class_num = int(class_num)
        width = int(width)
        height = int(height)
        data_pos = np.load(file_path)['pos'].astype(np.float64)  # num_events, triplet (x, y, t)
        data_neg = np.load(file_path)['neg'].astype(np.float64)  # num_events, triplet (x, y, t)

        dy = (254 - height) // 2
        dx = (254 - width) // 2
        input = torch.zeros([self.nsteps, 2, 256, 256])  # nsteps, in_channels=2, width=256, height=256

        x = data_pos[:, 0] + dx 
        y = data_pos[:, 1] + dy 
        t = (data_pos[:, 2] - 1) // (8 // self.nsteps)
        input[t, 0, x, y] = 1 

        x = data_neg[:, 0] + dx
        y = data_neg[:, 1] + dy
        t = (data_neg[:, 2] - 1) // (8 // self.nsteps)
        input[t, 0, x, y] = 1

        input = input[..., 16:240, 16:240]
        return input, torch.Tensor([class_num]).squeeze().int()

    def __len__(self):
        return self.__len__
    
    def split(self, ratio: float, random_split: bool = True):

        label_idx = []
        for i in range(1000):
            label_idx.append([])

        for i in range(self.__len__):
            if self.train:
                line = linecache.getline(self.train_label_path, i + 1)
            else:
                line = linecache.getline(self.val_label_path, i + 1)
            _, class_num, _, _ = line.split()
            class_num = int(class_num)
            label_idx[class_num].append(i)
        
        train_idx = []
        test_idx = []
        if random_split:
            for i in range(1000):
                np.random.shuffle(label_idx[i])

        for i in range(1000):
            pos = math.ceil(label_idx[i].__len__() * ratio)
            train_idx.extend(label_idx[i][0: pos])
            test_idx.extend(label_idx[i][pos: label_idx[i].__len__()])

        return torch.utils.data.Subset(self, train_idx), torch.utils.data.Subset(self, test_idx)

