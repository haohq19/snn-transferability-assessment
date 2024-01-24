import torch
import numpy as np
from functions.logME import logME
from functions.ApproxME import ApproxME


def assess(model, data_loader, mode='None'):
    # forward propagation and concatenate all outputs
    model.to('cuda')
    feature_maps = []
    labels = []
    for input, label in data_loader:
        input = input.cuda()
        with torch.no_grad():
            model(input)
        feature_map = model.feature
        feature_map = feature_map.mean(dim=1).cpu().detach().numpy()
        feature_maps.append(feature_map)
        label = label.cpu().detach().numpy()
        labels.append(label)
    feature_maps = np.concatenate(feature_maps, axis=0)
    labels = np.concatenate(labels, axis=0)
    model.to('cpu')
    # calculate model evidence
    if mode == 'LogME':
        score = logME(feature_maps, labels)
    elif mode == 'ApproxME':
        score = ApproxME(feature_maps, labels)
    else:
        raise NotImplementedError
    print(score)
    return score

def rank(models, data_loader, mode='None'):
    scores = []
    for model in models:
        score = assess(model, data_loader, mode)
        scores.append(score)
    return scores
