import os
import torch
import torch.nn as nn
import torchvision
import argparse
import spiking_resnet, sew_resnet
from torchvision import transforms
from torch.utils.data import DataLoader
from spikingjelly.clock_driven import functional


parser = argparse.ArgumentParser(description='PyTorch ImageNet Evaluation')
parser.add_argument('--device', default='cuda:0', help='device')
parser.add_argument('--data_path', default='/home/haohq/datasets/ImageNet', type=str, help='path to dataset')
parser.add_argument('--batch_size', default=512, type=int, help='batch size')
parser.add_argument('--model', default='sew_resnet18', type=str, help='model type (default: sew_resnet18)')
parser.add_argument('--nsteps', default=4, type=int, help='time steps')
parser.add_argument('--connect_f', default='ADD', type=str, help='spike-element-wise connect function')
parser.add_argument('--save_path', default='imagenet/weights/sew18_checkpoint_319.pth', type=str, help='path to saved weights')
parser.add_argument('--cache_dataset', help="cache dataset with normalization applied", action="store_true")
args = parser.parse_args()
print(args)

def _get_cache_path(filepath):
    import hashlib
    h = hashlib.sha1(filepath.encode()).hexdigest()
    cache_path = os.path.join("~", ".torchvision", "datasets", "imagefolder", h[:10] + ".pt")
    cache_path = os.path.expanduser(cache_path)
    return cache_path

def load_val_data(val_dir, cache_dataset):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    cache_path = _get_cache_path(val_dir)
    if cache_dataset and os.path.exists(cache_path):
        val_dataset, _ = torch.load(cache_path)
    else:
        val_dataset = torchvision.datasets.ImageFolder(
            val_dir,
            transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ]))
        if cache_dataset:
            os.makedirs(os.path.dirname(cache_path))
            torch.save((val_dataset, val_dir), cache_path)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    return val_dataset, val_dataloader

def load_model(args):
    # model
    if args.model in sew_resnet.__dict__:
        model = sew_resnet.__dict__[args.model](T=args.nsteps, connect_f=args.connect_f)
    elif args.model in spiking_resnet.__dict__:
        model = spiking_resnet.__dict__[args.model](T=args.nsteps)
    else:
        raise NotImplementedError(args.model)
    
    # load weights
    save = torch.load(args.save_path)
    weight = save['model']
    model.load_state_dict(weight)

    return model

def evaluate(args):

    # device
    device = torch.device(args.device)

    # data
    val_dir = os.path.join(args.data_path, 'val')
    _, val_dataloader = load_val_data(val_dir, args.cache_dataset)

    # model
    model = load_model(args)
    model.to(device)

    # evaluate
    model.eval()
    with torch.no_grad():
        top1_correct = 0
        top5_correct = 0
        count = 0
        total = len(val_dataloader.dataset)
        for input, target in val_dataloader:
            input = input.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            output = model(input)  # batch_size, num_classes
            functional.reset_net(model)

            # calculate the top5 and top1 accurate numbers
            _, predicted = output.topk(5, 1, True, True)  # batch_size, topk(5) 
            top1_correct += predicted[:, 0].eq(target).sum().item()
            top5_correct += predicted.T.eq(target[None]).sum().item()
            count += target.size(0)
            # print the process of evaluate with format 'calculated/total'
            print('Evaluating: {}/{}'.format(count, total), end='\r')

        top1_accuracy = top1_correct / total * 100
        top5_accuracy = top5_correct / total * 100

        print('Top1 accuracy: {:.2f}%'.format(top1_accuracy))
        print('Top5 accuracy: {:.2f}%'.format(top5_accuracy))

    

if __name__ == '__main__':
    evaluate(args)
