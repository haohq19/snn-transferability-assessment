# assess transferability of SNN

import os
import torch
import numpy as np
from functions.logME import logME
from functions.ApproxME import ApproxME
from functions.NCE import NCE
from functions.LEEP import LEEP


def compare_iter_with_cache(cache_dir):
    # compare the iteration number of logME and ApproxME with cached features
    # return score of logME, ApproxME, iter of logME, iter of ApproxME
    feature_maps = np.load(os.path.join(cache_dir, 'train_features.npy'))  # (N, T, D)
    feature_maps = feature_maps.mean(axis=1)                               # (N, D)
    labels = np.load(os.path.join(cache_dir, 'train_labels.npy'))          # (N,)
    score1, iter1 = logME(feature_maps, labels)
    score2, iter2 = ApproxME(feature_maps, labels)
    return score1, score2, iter1, iter2


def assess_with_cache(cache_dir):
    # calculate NCE, LEEP, logME with cached features and outputs
    # return score of NCE, LEEP, logME
    feature_maps = np.load(os.path.join(cache_dir, 'train_features.npy'))   # (N, T, D)
    feature_maps = feature_maps.mean(axis=1)                                # (N, D)
    labels = np.load(os.path.join(cache_dir, 'train_labels.npy'))           # (N,)
    pseudo_logits = np.load(os.path.join(cache_dir, 'train_logits.npy'))    # (N, C)
    pseudo_labels = np.argmax(pseudo_logits, axis=1)                        # (N,)

    score_nce = NCE(pseudo_labels, labels)
    score_leep = LEEP(pseudo_logits, labels)
    score_logme, _ = logME(feature_maps, labels)
    return score_nce, score_leep, score_logme