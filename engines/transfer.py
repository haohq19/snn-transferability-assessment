import os
import numpy as np
import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, TensorDataset
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
            
            if len(data.shape) == 5:  # event-based data
                # data.shape = [batch_size, nsteps, channels, height, width]
                # input.shape = [nsteps, batch_size, channels, height, width]
                input = data.transpose(0, 1).cuda(non_blocking=True)    
            
            elif len(data.shape) == 4:  # static data
                # data.shape = [batch_size, channels, height, width]
                # input.shape = [batch_size, channels, height, width]
                input = data.cuda(non_blocking=True)
            else:
                raise ValueError('Invalid data shape')

            # label.shape = [batch_size]
            label = label.numpy()                                   # label.shape = [batch_size]

            output = model(input).mean(dim=0)                       # output.shape = [batch_size, num_classes]
            
            feature_map = model.feature.detach().cpu().numpy()      # feature_map.shape = [nsteps, batch_size, num_features]
            features.append(feature_map)
            logit = output.softmax(dim=1).detach().cpu().numpy()    # logit.shape = [batch_size, num_classes]
            logits.append(logit)
            labels.append(label)

            functional.reset_net(model)

            process_bar.update(1)
        process_bar.close()
        
        features = np.concatenate(features, axis=1).transpose(1, 0, 2)  # features.shape = [nsamples, nsteps, num_features]
        logits = np.concatenate(logits, axis=0)                         # logits.shape = [nsamples, num_classes]
        labels = np.concatenate(labels, axis=0)                         # labels.shape = [nsamples]
        np.save(os.path.join(cache_dir, 'train_features.npy'), features)
        np.save(os.path.join(cache_dir, 'train_logits.npy'), logits)
        np.save(os.path.join(cache_dir, 'train_labels.npy'), labels)

        # valid_loader
        features = []
        labels = []
        nsteps_per_epoch = len(valid_loader)
        process_bar = tqdm.tqdm(total=nsteps_per_epoch)
        for data, label in valid_loader:
            if len(data.shape) == 5:
                input = data.transpose(0, 1).cuda(non_blocking=True)    
            
            elif len(data.shape) == 4:
                input = data.cuda(non_blocking=True)
            else:
                raise ValueError('Invalid data shape')

            # label.shape = [batch_size]
            label = label.numpy()                                   # label.shape = [batch_size]

            model(input)
            
            feature_map = model.feature.detach().cpu().numpy()
            features.append(feature_map)
            labels.append(label)

            functional.reset_net(model)

            process_bar.update(1)
        process_bar.close()

        features = np.concatenate(features, axis=1).transpose(1, 0, 2)
        labels = np.concatenate(labels, axis=0)
        np.save(os.path.join(cache_dir, 'valid_features.npy'), features)
        np.save(os.path.join(cache_dir, 'valid_labels.npy'), labels)

        # test_loader
        features = []
        labels = []
        nsteps_per_epoch = len(test_loader)
        process_bar = tqdm.tqdm(total=nsteps_per_epoch)
        for data, label in test_loader:
            if len(data.shape) == 5:
                input = data.transpose(0, 1).cuda(non_blocking=True)
            elif len(data.shape) == 4:
                input = data.cuda(non_blocking=True)
            else:
                raise ValueError('Invalid data shape')

            model(input)
            
            feature_map = model.feature.detach().cpu().numpy()
            features.append(feature_map)
            labels.append(label)

            functional.reset_net(model)

            process_bar.update(1)
        process_bar.close()

        features = np.concatenate(features, axis=1).transpose(1, 0, 2)
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
    output_dir: str,
):  
    
    tb_writer = SummaryWriter(output_dir + '/log')
    print('Save logs to [{}]'.format(output_dir + '/log'))

    # best model
    best_model = None
    best_acc = 0
    best_acc_epoch = 0

    for epoch in range(nepochs):
        print('Epoch [{}/{}]'.format(epoch+1, nepochs))

        # train
        model.train()

        # accuracy
        top1_correct = 0
        top5_correct = 0
        nsamples_per_epoch = len(train_loader.dataset)
        epoch_loss = 0

        # process bar
        nsteps_per_epoch = len(train_loader)
        process_bar = tqdm.tqdm(total=nsteps_per_epoch)

        for step, (data, label) in enumerate(train_loader):
            # data.shape = [batch_size, nsteps, num_features]
            # label.shape = [batch_size]

            input = data.transpose(0, 1).cuda(non_blocking=True)                    # input.shape = [nsteps, batch_size, num_features]
            target = F.one_hot(label, num_classes).float().cuda(non_blocking=True)  # target.shape = [batch_size, num_classes]
            
            output = model(input)   # output.shape = [batch_size, num_classes]
            loss = criterion(output, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            functional.reset_net(model)

            # calculate accuracy
            _, predicted = output.cpu().topk(5, 1, True, True)
            top1_correct += predicted[:, 0].eq(label).sum().item()
            top5_correct += predicted.T.eq(label[None]).sum().item()

            # process bar
            process_bar.update(1)

            # tensorboard
            tb_writer.add_scalar('step/loss', loss.item(), epoch * nsteps_per_epoch + step)
            
            epoch_loss += loss.item() * input.shape[1]
            
        top1_accuracy = top1_correct / nsamples_per_epoch
        top5_accuracy = top5_correct / nsamples_per_epoch
        
        tb_writer.add_scalar('train/acc@1', top1_accuracy, epoch + 1)
        tb_writer.add_scalar('train/acc@5', top5_accuracy, epoch + 1)
        tb_writer.add_scalar('train/avg_loss', epoch_loss / nsamples_per_epoch , epoch + 1)
        
        process_bar.close()
        print('train || acc@1: {:.5f}, acc@5: {:.5f}, avg_loss: {:.6f}, cor@1: {}, cor@5: {}, total: {}'.format(
             top1_accuracy, top5_accuracy, epoch_loss / nsamples_per_epoch, top1_correct, top5_correct, nsamples_per_epoch
            )
        )

        # valid
        model.eval()
        top1_correct = 0
        top5_correct = 0
        nsamples_per_epoch = len(valid_loader.dataset)
        epoch_loss = 0

        with torch.no_grad():
            for step, (data, label) in enumerate(valid_loader):
                # data.shape = [batch_size, nsteps, num_features]
                # label.shape = [batch_size]

                input = data.transpose(0, 1).cuda(non_blocking=True)                    # input.shape = [nsteps, batch_size, num_features]
                target = F.one_hot(label, num_classes).float().cuda(non_blocking=True)  # target.shape = [batch_size, num_classes]

                output = model(input)
                loss = criterion(output, target)
                
                functional.reset_net(model)

                # calculate accuracy
                _, predicted = output.cpu().topk(5, 1, True, True)  # batch_size, topk(5) 
                top1_correct += predicted[:, 0].eq(label).sum().item()
                top5_correct += predicted.T.eq(label[None]).sum().item()
                epoch_loss += loss.item() * input.shape[1]

        top1_accuracy = top1_correct / nsamples_per_epoch
        top5_accuracy = top5_correct / nsamples_per_epoch

        tb_writer.add_scalar('valid/acc@1', top1_accuracy, epoch + 1)
        tb_writer.add_scalar('valid/acc@5', top5_accuracy, epoch + 1)
        tb_writer.add_scalar('valid/avg_loss', epoch_loss / nsamples_per_epoch, epoch + 1)
        print('valid || acc@1: {:.5f}, acc@5: {:.5f}, avg_loss: {:.6f}, cor@1: {}, cor@5: {}, total: {}'.format(
             top1_accuracy, top5_accuracy, epoch_loss / nsamples_per_epoch, top1_correct, top5_correct, nsamples_per_epoch
            )
        )

        # best model info
        if top1_accuracy >= best_acc:
            best_acc = top1_accuracy
            best_model = model
            best_acc_epoch = epoch
    
    # save best model
    checkpoint = {
        'model': best_model.state_dict(),
    }
    save_name = 'checkpoints/best_{}_{}.pth'.format(best_acc_epoch, best_acc)
    torch.save(checkpoint, os.path.join(output_dir, save_name))
    print('Save best model to [{}]'.format(output_dir))

    print('Best valid acc@1: {}'.format(best_acc))
    return best_model, best_acc


def test(
    model: nn.Module,
    test_loader: DataLoader,
):
    top1_correct = 0
    top5_correct = 0
    nsamples_per_epoch = len(test_loader.dataset)
    with torch.no_grad():
        for step, (data, label) in enumerate(test_loader):
            # data.shape = [batch_size, nsteps, num_features]
            # label.shape = [batch_size]

            input = data.transpose(0, 1).cuda(non_blocking=True)   # input.shape = [nsteps, batch_size, num_features]

            output = model(input)

            functional.reset_net(model)

            # calculate the top5 and top1 accurate numbers
            _, predicted = output.cpu().topk(5, 1, True, True)
            top1_correct += predicted[:, 0].eq(label).sum().item()
            top5_correct += predicted.T.eq(label[None]).sum().item()

    top1_accuracy = top1_correct / nsamples_per_epoch
    top5_accuracy = top5_correct / nsamples_per_epoch

    print('test || acc@1: {:.5f}, acc@5: {:.5f}, cor@1: {}, cor@5: {}, total: {}'.format(
             top1_accuracy, top5_accuracy, top1_correct, top5_correct, nsamples_per_epoch
            )
        )
    
    return top1_accuracy