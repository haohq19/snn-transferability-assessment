# # assess transferability of SNN on static datasets

import torch
import numpy as np
from functions.logME import logME
from functions.ApproxME import ApproxME
from functions.NCE import NCE
from functions.LEEP import LEEP


def compare_iter(model, dataloader):
    # compare the iteration number of logME and ApproxME
    # return score of logME, ApproxME, iter of logME, iter of ApproxME
    model.to('cuda')
    feature_maps = []
    labels = []
    for input, label in dataloader:
        input = input.cuda()
        with torch.no_grad():
            model(input)
        # for static dataset model.feature: (T, N, D)
        feature_map = model.feature.mean(dim=0).cpu().detach().numpy()  # (N, D) 
        feature_maps.append(feature_map)
        label = label.cpu().detach().numpy()
        labels.append(label)
    feature_maps = np.concatenate(feature_maps, axis=0)  # (N, D)
    labels = np.concatenate(labels, axis=0)
    model.to('cpu')
    score1, iter1 = logME(feature_maps, labels)
    score2, iter2 = ApproxME(feature_maps, labels)
    return score1, score2, iter1, iter2


def compare_iter_with_cache(cache_dir):
    # compare the iteration number of logME and ApproxME with cached features
    # return score of logME, ApproxME, iter of logME, iter of ApproxME
    feature_maps = np.load(cache_dir + '/train_features.npy')  # (N, T, D)
    feature_maps = feature_maps.mean(axis=1)  # (N, D)
    labels = np.load(cache_dir + '/train_labels.npy')  # (N,)
    score1, iter1 = logME(feature_maps, labels)
    score2, iter2 = ApproxME(feature_maps, labels)
    return score1, score2, iter1, iter2


def assess_all(model, dataloader):
    # assess NCE, LEEP, logME
    # return score of NCE, LEEP, logME
    model.to('cuda')
    pseudo_labels = []  # NCE
    pseudo_logits = []  # LEEP
    feature_maps = []  # logME
    labels = []
    for input, label in dataloader:
        input = input.cuda()
        with torch.no_grad():
            output = model(input)  # (N, C)
            output = output.softmax(dim=1)  # (N, C)
            output = output.cpu().detach().numpy()
        # NCE
        pseudo_label = np.argmax(output, axis=1)
        pseudo_labels.append(pseudo_label)
        # LEEP
        pseudo_logits.append(output)
        # logME
        feature_map = model.feature.mean(dim=0).cpu().detach().numpy()
        feature_maps.append(feature_map)
        # label
        label = label.cpu().detach().numpy()
        labels.append(label)
    pseudo_labels = np.concatenate(pseudo_labels, axis=0)
    pseudo_logits = np.concatenate(pseudo_logits, axis=0)
    feature_maps = np.concatenate(feature_maps, axis=0)
    labels = np.concatenate(labels, axis=0)
    model.to('cpu')
    score1 = NCE(pseudo_labels, labels)
    score2 = LEEP(pseudo_logits, labels)
    score3, _ = logME(feature_maps, labels)
    return score1, score2, score3


def assess_all_with_cache(cache_dir):
    # assess NCE, LEEP, logME with cached features and outputs
    # return score of NCE, LEEP, logME
    feature_maps = np.load(cache_dir + '/train_features.npy')  # (N, T, D)
    feature_maps = feature_maps.mean(axis=1)  # (N, D)
    labels = np.load(cache_dir + '/train_labels.npy')  # (N,)
    pseudo_logits = np.load(cache_dir + '/train_logits.npy')  # (N, C)
    pseudo_labels = np.argmax(pseudo_logits, axis=1)  # (N,)

    score1 = NCE(pseudo_labels, labels)
    score2 = LEEP(pseudo_logits, labels)
    score3, _ = logME(feature_maps, labels)
    return score1, score2, score3