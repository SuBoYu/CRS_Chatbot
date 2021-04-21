# BB-8 and R2-D2 are best friends.

import sys
sys.path.insert(0, '../FM')
sys.path.insert(0, '../yelp')

import random
import torch
import torch.nn as nn
import json
import pickle
import numpy as np
from sklearn.metrics import roc_auc_score
import time
from torch.nn.utils import clip_grad_value_, clip_grad_norm_
from torch.nn.utils.rnn import pad_sequence
from config import global_config as cfg
from sklearn.metrics.pairwise import cosine_similarity
import argparse
import sys

from FM_old import FactorizationMachine


def feature_similarity(given_preference, userID, TopKTaxo):
    preference_matrix_all = cfg.user_emb[[userID]]
    if len(given_preference) > 0:
        preference_matrix = cfg.emb_matrix[given_preference]
        preference_matrix_all = np.concatenate((preference_matrix, preference_matrix_all), axis=0)

    result_dict = dict()

    for index, big_feature in enumerate(cfg.FACET_POOL):
        big_feature_matrix = cfg.emb_matrix[index, :]
        big_feature_matrix = big_feature_matrix.reshape(-1, 64)
        cosine_result = cosine_similarity(big_feature_matrix, preference_matrix_all)
        cosine_result = np.sum(cosine_result, axis=1)
        normalize_factor = 5.0  #We roughly normalize the feature similarity to a scale similar to max-entropy scores. Followers of our work can try more sophisticated factors.
        result_dict[big_feature] = normalize_factor * float(cosine_result[0]) / len(given_preference)
    return result_dict
