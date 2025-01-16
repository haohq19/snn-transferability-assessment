# pretrain spiking on ES-ImageNet

import os
import tqdm
import argparse
import numpy as np
import random
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from spikingjelly.activation_based import functional
from utils.dist import is_master, save_on_master, global_meters_all_sum

_seed_ = 2020
random.seed(2020)
torch.manual_seed(_seed_)
torch.cuda.manual_seed_all(_seed_)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(_seed_)

def train(
    model: nn.Module,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    train_loader: DataLoader,
    val_loader: DataLoader,
    nepochs: int,
    epoch: int,
    output_dir: str,
    args: argparse.Namespace,
):  
    if is_master():
        tb_writer = SummaryWriter(output_dir + '/log')
    print('Save log to {}'.format(output_dir + '/log'))

    torch.cuda.empty_cache()
    
    epoch = epoch
    while(epoch < nepochs):
        print('Epoch [{}/{}]'.format(epoch+1, nepochs))
        
         # train 
        model.train()
        top1_correct = 0
        top5_correct = 0
        nsamples_per_epoch = len(train_loader.dataset)
        epoch_total_loss = 0
        nsteps_per_epoch = len(train_loader)

        # progress bar
        if is_master():
            process_bar = tqdm.tqdm(total=nsteps_per_epoch)

        for step, (data, label) in enumerate(train_loader):
            
            input = data.transpose(0, 1).cuda(non_blocking=True)                    # input.shape = [nsteps, batch_size, channel, height, width]
            target = label.cuda(non_blocking=True)                                  # target.shape = [batch_size]
            
            output = model(input)                                                   # output.shape = [batch_size, num_classes]
            loss = criterion(output, target)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            functional.reset_net(model)

            # calculate the top5 and top1 accurate numbers
            _, predicted = output.cpu().topk(5, 1, True, True)
            top1_correct += predicted[:, 0].eq(label).sum().item()
            top5_correct += predicted.T.eq(label[None]).sum().item()
            epoch_total_loss += loss.item()

            if is_master():
                tb_writer.add_scalar('step/loss', loss.item(), epoch * nsteps_per_epoch + step)
                process_bar.update(1)
        
        if args.distributed:
            top1_correct, top5_correct, epoch_total_loss = global_meters_all_sum(args, top1_correct, top5_correct, epoch_total_loss)
        top1_accuracy = top1_correct / nsamples_per_epoch
        top5_accuracy = top5_correct / nsamples_per_epoch

        if is_master():    
            tb_writer.add_scalar('train/acc@1', top1_accuracy, epoch + 1)
            tb_writer.add_scalar('train/acc@5', top5_accuracy, epoch + 1)
            tb_writer.add_scalar('train/avg_loss', epoch_total_loss / nsamples_per_epoch, epoch + 1)
            process_bar.close()

        print('train || acc@1: {:.5f}, acc@5: {:.5f}, avg_loss: {:.6f}, cor@1: {}, cor@5: {}, total: {}'.format(
             top1_accuracy, top5_accuracy, epoch_total_loss / nsamples_per_epoch, top1_correct, top5_correct, nsamples_per_epoch
            )
        )
        
        # validate
        model.eval()
        top1_correct = 0
        top5_correct = 0
        nsamples_per_epoch = len(val_loader.dataset)
        epoch_total_loss = 0
        nsteps_per_epoch = len(val_loader)
        if is_master():
            process_bar = tqdm.tqdm(total=nsteps_per_epoch)
        with torch.no_grad():
            for step, (data, label) in enumerate(val_loader):

                input = data.transpose(0, 1).cuda(non_blocking=True)
                target = label.cuda(non_blocking=True)

                output = model(input)
                loss = criterion(output, target)

                functional.reset_net(model)

                # calculate the top5 and top1 accurate numbers
                _, predicted = output.cpu().topk(5, 1, True, True)  # batch_size, topk(5) 
                top1_correct += predicted[:, 0].eq(label).sum().item()
                top5_correct += predicted.T.eq(label[None]).sum().item()
                epoch_total_loss += loss.item()
                if is_master():
                    process_bar.update(1)
        
        if args.distributed:
            top1_correct, top5_correct, epoch_total_loss = global_meters_all_sum(args, top1_correct, top5_correct, epoch_total_loss)
        top1_accuracy = top1_correct / nsamples_per_epoch
        top5_accuracy = top5_correct / nsamples_per_epoch
        if is_master():   
            tb_writer.add_scalar('val/acc@1', top1_accuracy, epoch + 1)
            tb_writer.add_scalar('val/acc@5', top5_accuracy, epoch + 1)
            tb_writer.add_scalar('val/avg_loss', epoch_total_loss / nsamples_per_epoch, epoch + 1)
            process_bar.close()

        print('valid || acc@1: {:.5f}, acc@5: {:.5f}, avg_loss: {:.6f}, cor@1: {}, cor@5: {}, total: {}'.format(
             top1_accuracy, top5_accuracy, epoch_total_loss / nsamples_per_epoch, top1_correct, top5_correct, nsamples_per_epoch
            )
        )
        
        
        # save
        epoch += 1
        scheduler.step()
        if epoch % args.save_freq == 0:
            checkpoint = {
                'model': model.module.state_dict() if args.distributed else model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'epoch': epoch,
            }
            save_name = 'checkpoints/checkpoint_epoch{}_acc{:.2f}.pth'.format(epoch, top1_accuracy)
            save_on_master(checkpoint, os.path.join(output_dir, save_name))
            print('Saved checkpoint to [{}]'.format(output_dir))