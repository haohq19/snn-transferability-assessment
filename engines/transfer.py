import os
import numpy as np
import tqdm
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.functional import one_hot
from spikingjelly.activation_based import functional

def cache_representations(
    model: nn.Module,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    test_loader: DataLoader,
    cache_dir: str,
):  
    if os.path.exists(cache_dir):
        print('Cache already exists')
        return
    else:
        os.makedirs(cache_dir)
        print('Cache not found, make cache directory at [{}]'.format(cache_dir))
    
    with torch.no_grad():
        model.eval()

        # train_loader
        features = []
        logits = []
        labels = []
        nsteps_per_epoch = len(train_loader)
        process_bar = tqdm.tqdm(total=nsteps_per_epoch)
        for data, label in train_loader:
            input = data.cuda(non_blocking=True)
            label = label.numpy()
            output = model(input).mean(dim=1)  # N, C
            
            feature_map = model.feature.detach().cpu().numpy()  # N, T, D
            features.append(feature_map)
            logit = output.softmax(dim=1).detach().cpu().numpy()  # N, C
            logits.append(logit)
            labels.append(label)

            functional.reset_net(model)

            process_bar.update(1)
        process_bar.close()
        
        features = np.concatenate(features, axis=0)  # N, T, D
        logits = np.concatenate(logits, axis=0) # N, C
        labels = np.concatenate(labels, axis=0)  # N
        np.save(os.path.join(cache_dir, 'train_features.npy'), features)
        np.save(os.path.join(cache_dir, 'train_logits.npy'), logits)
        np.save(os.path.join(cache_dir, 'train_labels.npy'), labels)

        # valid_loader
        features = []
        labels = []
        nsteps_per_epoch = len(valid_loader)
        process_bar = tqdm.tqdm(total=nsteps_per_epoch)
        for data, label in valid_loader:
            input = data.cuda(non_blocking=True)
            label = label.numpy()
            model(input)
            
            feature_map = model.feature.detach().cpu().numpy()
            features.append(feature_map)
            labels.append(label)

            functional.reset_net(model)

            process_bar.update(1)
        process_bar.close()

        features = np.concatenate(features, axis=0)
        labels = np.concatenate(labels, axis=0)
        np.save(os.path.join(cache_dir, 'valid_features.npy'), features)
        np.save(os.path.join(cache_dir, 'valid_labels.npy'), labels)

        # test_loader
        features = []
        labels = []
        nsteps_per_epoch = len(test_loader)
        process_bar = tqdm.tqdm(total=nsteps_per_epoch)
        for data, label in test_loader:
            input = data.cuda(non_blocking=True)
            label = label.numpy()
            model(input)
            
            feature_map = model.feature.detach().cpu().numpy()
            features.append(feature_map)
            labels.append(label)

            functional.reset_net(model)

            process_bar.update(1)
        process_bar.close()

        features = np.concatenate(features, axis=0)
        labels = np.concatenate(labels, axis=0)
        np.save(os.path.join(cache_dir, 'test_features.npy'), features)
        np.save(os.path.join(cache_dir, 'test_labels.npy'), labels)

        print('Cached feature map saved to [{}]'.format(os.path.join(cache_dir)))


def get_data_loader_from_cached_representations(args):
    train_dataset = TensorDataset(torch.from_numpy(np.load(os.path.join(args.cache_dir, 'train_features.npy'))), torch.from_numpy(np.load(os.path.join(args.cache_dir, 'train_labels.npy'))))
    valid_dataset = TensorDataset(torch.from_numpy(np.load(os.path.join(args.cache_dir, 'valid_features.npy'))), torch.from_numpy(np.load(os.path.join(args.cache_dir, 'valid_labels.npy'))))
    test_dataset = TensorDataset(torch.from_numpy(np.load(os.path.join(args.cache_dir, 'test_features.npy'))), torch.from_numpy(np.load(os.path.join(args.cache_dir, 'test_labels.npy'))))
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.nworkers, pin_memory=True, drop_last=False)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.nworkers, pin_memory=True, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.nworkers, pin_memory=True, drop_last=False)
    return train_loader, valid_loader, test_loader



def train(
    model: nn.Module,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    num_classes: int,
    nepochs: int,
    epoch: int,
    output_dir: str,
):  
    
    tb_writer = SummaryWriter(output_dir + '/log')
    print('Save logs to [{}]'.format(output_dir + '/log'))

    # train 
    epoch = epoch
    best_model = None
    best_acc = 0

    while(epoch < nepochs):
        print('Epoch [{}/{}]'.format(epoch+1, nepochs))
        model.train()
        top1_correct = 0
        # top5_correct = 0
        nsamples_per_epoch = len(train_loader.dataset)
        epoch_loss = 0
        nsteps_per_epoch = len(train_loader)
        process_bar = tqdm.tqdm(total=nsteps_per_epoch)
        
        for step, (data, label) in enumerate(train_loader):

            input = data.cuda(non_blocking=True)
            target = one_hot(label, num_classes).cuda(non_blocking=True)
            
            input = input.transpose(0, 1)  # T, N, D
            output = model(input)
            loss = criterion(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            functional.reset_net(model)

            # calculate the top5 and top1 accurate numbers
            _, predicted = output.cpu().topk(5, 1, True, True)
            top1_correct += predicted[:, 0].eq(label).sum().item()
            top5_correct += predicted.T.eq(label[None]).sum().item()

            process_bar.update(1)
            tb_writer.add_scalar('step_loss', loss.item(), epoch * nsteps_per_epoch + step)
            
            epoch_loss += loss.item() * input.shape[1]
            
        top1_accuracy = top1_correct / nsamples_per_epoch
        top5_accuracy = top5_correct / nsamples_per_epoch
        
        tb_writer.add_scalar('train/acc@1', top1_accuracy, epoch + 1)
        tb_writer.add_scalar('train/acc@5', top5_accuracy, epoch + 1)
        tb_writer.add_scalar('train/loss', epoch_loss / nsamples_per_epoch , epoch + 1)
        
        process_bar.close()
        print('train || acc@1: {:.5f}, acc@5: {:.5f}, avg_loss: {:.6f}, cor@1: {}, cor@5: {}, total: {}'.format(
             top1_accuracy, top5_accuracy, epoch_loss / nsamples_per_epoch, top1_correct, top5_correct, nsamples_per_epoch
            )
        )

        # evaluate
        model.eval()
        top1_correct = 0
        top5_correct = 0
        nsamples_per_epoch = len(valid_loader.dataset)
        epoch_loss = 0
        with torch.no_grad():
            for data, label in valid_loader:
                input = data.cuda(non_blocking=True)
                label = label.cuda(non_blocking=True)
                target = one_hot(label, num_classes).cuda(non_blocking=True)
                input = input.transpose(0, 1)  # T, N, D
                output = model(input)
                loss = criterion(output, target)
                functional.reset_net(model)

                # calculate the top5 and top1 accurate numbers
                _, predicted = output.cpu().topk(5, 1, True, True)  # batch_size, topk(5) 
                top1_correct += predicted[:, 0].eq(label).sum().item()
                top5_correct += predicted.T.eq(label[None]).sum().item()
                epoch_loss += loss.item() * input.shape[1]

        top1_accuracy = top1_correct / nsamples_per_epoch
        top5_accuracy = top5_correct / nsamples_per_epoch

        tb_writer.add_scalar('valid_acc@1', top1_accuracy, epoch + 1)
        tb_writer.add_scalar('valid_acc@5', top5_accuracy, epoch + 1)
        tb_writer.add_scalar('valid_loss', epoch_loss / nsamples_per_epoch, epoch + 1)
        print('valid || acc@1: {:.5f}, acc@5: {:.5f}, avg_loss: {:.6f}, cor@1: {}, cor@5: {}, total: {}'.format(
             top1_accuracy, top5_accuracy, epoch_loss / nsamples_per_epoch, top1_correct, top5_correct, nsamples_per_epoch
            )
        )
        epoch += 1

        # save best
        if top1_accuracy >= best_acc:
            best_acc = top1_accuracy
            best_model = model
            checkpoint = {
                'model': best_model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
            }
            save_name = 'checkpoint/best_{}.pth'.format(top1_accuracy)
            torch.save(checkpoint, os.path.join(output_dir, save_name))
            print('Save best model to [{}]'.format(output_dir))
    print('best_valid_acc@1: {}'.format(best_acc))
    return best_model


def test(
    model: nn.Module,
    test_loader: DataLoader,
    output_dir: str,
):
    top1_correct = 0
    top5_correct = 0
    nsamples_per_epoch = len(test_loader.dataset)
    with torch.no_grad():
        for data, label in test_loader:
            input = data.cuda(non_blocking=True)
            input = input.transpose(0, 1)  # T, N, D
            output = model(input)
            functional.reset_net(model)

            # calculate the top5 and top1 accurate numbers
            _, predicted = output.cpu().topk(5, 1, True, True)
            top1_correct += predicted[:, 0].eq(label).sum().item()
            top5_correct += predicted.T.eq(label[None]).sum().item()

    top1_accuracy = top1_correct / nsamples_per_epoch
    top5_accuracy = top5_correct / nsamples_per_epoch

    print('test || acc@1: {:.5f}, acc@5: {:.5f},cor@1: {}, cor@5: {}, total: {}'.format(
             top1_accuracy, top5_accuracy, top1_correct, top5_correct, nsamples_per_epoch
            )
        )