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

from collections import defaultdict

import argparse
import sys
from heapq import nlargest, nsmallest

from FM_old import FactorizationMachine
from config import global_config as cfg


def cuda_(var):
    return var.cuda() if torch.cuda.is_available() else var


def rank_items(given_preference, userID, itemID, skip_big_feature, FM_model, candidate_list, write_fp, after_update):
    FM_model.eval()
    start = time.time()
    mini_ui_pair = np.zeros((len(candidate_list), 2))
    for index, itemID in enumerate(candidate_list):
        mini_ui_pair[index, :] = [userID, itemID + len(cfg.user_list)]
    mini_ui_pair = torch.from_numpy(mini_ui_pair).long()
    mini_ui_pair = cuda_(mini_ui_pair)

    static_preference_index = torch.LongTensor(given_preference).expand(len(candidate_list), len(given_preference))  # candidate_list, given preference
    static_preference_index = cuda_(static_preference_index)
    static_score, _, _ = FM_model(mini_ui_pair, None, static_preference_index)
    static_score = static_score.detach().cpu().numpy()

    score_dict = dict()
    for index, item in enumerate(static_score.reshape(-1).tolist()):
        score_dict[item] = candidate_list[index]

    assert len(candidate_list) == len(static_score)
    assert candidate_list[-1] == itemID
    target_value = static_score[-1]
    ranked_score = nlargest(100000, static_score.reshape(-1).tolist())
    target_position = ranked_score.index(target_value)
    top10_mean = np.mean(np.array(ranked_score[: 10]))

    ranked_item = [score_dict[score] for score in ranked_score]  # It is the ranked items to return

    predictions = static_score.reshape((len(static_score), 1)[0])
    y_true = [0] * len(predictions)
    y_true[-1] = 1
    tmp = list(zip(y_true, predictions))
    tmp = sorted(tmp, key=lambda x: x[1], reverse=True)
    y_true, predictions = zip(*tmp)
    if len(y_true) > 1:
        auc = roc_auc_score(y_true, predictions)
        # with open(write_fp, 'a') as f:
        #     if after_update == 1:
        #         f.write('AFTER UPDATE Target position is: {} in {}, AUC: {}\n'.format(target_position, len(ranked_score), auc))
        #         f.write('Tail 10s are: {}\n'.format(ranked_item[-10:]))
        #     else:
        #         f.write('Target position is: {} in {}, AUC: {}\n'.format(target_position, len(ranked_score), auc))
    FM_model.train()

    return ranked_item
