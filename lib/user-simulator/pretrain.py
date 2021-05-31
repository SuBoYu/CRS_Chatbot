import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import pickle
import numpy as np
from torch.nn import functional as F
import time
from torch.autograd import gradcheck
import argparse

from tqdm import  tqdm
from config import global_config as cfg
from utils_entropy import cal_ent
from pn import PolicyNetwork
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import glob
import os
from time import sleep

random.seed(1)
torch.cuda.manual_seed_all(1)
torch.manual_seed(1)

def cuda_(var):
    return var.cuda() if torch.cuda.is_available() else var

def vectorize(feat, recent_candidate_list):

    c = cal_ent(recent_candidate_list)
    entropy_dict = c.do_job()

    list3 = [v for k, v in entropy_dict.items()]
    #list4 = [v for k, v in sim_dict2.items()]
    list4 = [0] * len(cfg.tag_map)

    #list3.remove(list3[-1]

    assert len(list3) == len(cfg.FACET_POOL)
    assert len(list4) == len(cfg.FACET_POOL)

    MAX_TURN = 5
    history_list = [1] * min(MAX_TURN-1, len(feat))
    list5 = history_list + [0] * (MAX_TURN - len(history_list))
    assert len(list5) == 5
    list6 = [0] * 8
    if len(recent_candidate_list) <= 10:
        list6[0] = 1
    if len(recent_candidate_list) > 10 and len(recent_candidate_list) <= 50:
        list6[1] = 1
    if len(recent_candidate_list) > 50 and len(recent_candidate_list) <= 100:
        list6[2] = 1
    if len(recent_candidate_list) > 100 and len(recent_candidate_list) <= 200:
        list6[3] = 1
    if len(recent_candidate_list) > 200 and len(recent_candidate_list) <= 300:
        list6[4] = 1
    if len(recent_candidate_list) > 300 and len(recent_candidate_list) <= 500:
        list6[5] = 1
    if len(recent_candidate_list) > 500 and len(recent_candidate_list) <= 1000:
        list6[6] = 1
    if len(recent_candidate_list) > 1000:
        list6[7] = 1

    list_cat = list3 + list4 + list5 + list6
    # list_cat = np.array(list_cat)
    #print(list_cat)

    #assert len(list_cat) == 89
    return list_cat


def validate(purpose, train_list, valid_list, test_list, model):
    model.eval()

    if purpose == 1:
        data = train_list#
    elif purpose == 2:
        data = valid_list
    else:
        data = test_list

    data = data
    bs = 256
    max_iter = int(len(data) / bs)
    start = time.time()
    epoch_loss = 0
    correct = 0

    y_true, y_pred = list(), list()
    for iter_ in range(max_iter):
        left, right = iter_ * bs, min(len(train_list), (iter_ + 1) * bs)
        data_batch = data[left: right]

        temp_out = np.array([item[0] for item in data_batch])

        a = [item[1] for item in data_batch]
        s = a[0].shape[0]

        b = np.concatenate(a).reshape(-1, s)

        temp_in = torch.from_numpy(b).float()
        temp_target = torch.from_numpy(temp_out).long()

        temp_in, temp_target = cuda_(temp_in), cuda_(temp_target)

        pred = model(temp_in)
        y_true += temp_out.tolist()
        pred_result = pred.data.max(1)[1]
        correct += sum(np.equal(pred_result.cpu().numpy(), temp_out))

        y_pred += pred_result.cpu().numpy().tolist()

    print('Validating purpose {} takes {} seconds, cumulative loss is: {}, accuracy: {}%'.format(purpose, time.time() - start, epoch_loss / max_iter, correct * 100.0 / (max_iter * bs)))
    print(classification_report(y_true, y_pred))
    model.train()


def train(bs, train_list, valid_list, test_list, optimizer, model, criterion, epoch, model_path):

    with open('../../data/FM-train-data/item_dict.json', 'r') as f:
        item_dict = json.load(f)

    # print('-------validating before training {} epoch-------'.format(epoch))
    # if epoch > 0:
    #     validate(1, train_list, valid_list, test_list, model)

    model.train()
    random.shuffle(train_list)
    epoch_loss = 0
    max_iter = int(len(train_list) / bs)
    start = time.time()
    SR = 0
    n = 0
    for iter_ in tqdm(range(max_iter)):
        left, right = iter_ * bs, min(len(train_list), (iter_ + 1) * bs)
        data_batch = train_list[left: right]

        temp_out = np.array([item[0][0] for item in data_batch])

        # 要改的地方

        b = []
        a = [item[1] for item in data_batch]
        s = len(cfg.tag_map)*2+cfg.MAX_TURN+8
        for f in a:
            feat_set = set(f)
            cand_item = []
            for item in item_dict:
                if len(set(item_dict[item]['feature_index']) & feat_set) == len(feat_set):
                    cand_item.append(item)
            b.append(vectorize(f, cand_item))

        b = np.array(b)
        # b = np.concatenate(b).reshape(-1, s)

        temp_in = torch.from_numpy(b).float()

        temp_target = torch.from_numpy(temp_out).long()

        temp_in, temp_target = cuda_(temp_in), cuda_(temp_target)

        pred = model(temp_in)
        _pred = torch.argmax(pred, dim=1)
        n += pred.shape[0]
        sr = [1 for i in range(len(_pred)) if _pred[i] == temp_target[i]]
        SR += np.sum(sr)
        loss = criterion(pred, temp_target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.data

        # if iter_ % 500 == 0:
        #     print('{} seconds to finished {}% cumulative loss is: {}'.format(time.time() - start, iter_ * 100.0 / max_iter, epoch_loss / iter_))
    print(epoch_loss)
    print(f"SR: {SR/n}")
    if epoch == 7:
        PATH = model_path
        torch.save(model.state_dict(), PATH)
        print('Model saved at {}'.format(PATH))
        return

def main():
    parser = argparse.ArgumentParser(description="Pretrain Policy Network")
    parser.add_argument('-inputdim', type=int, dest='inputdim', help='input dimension')
    parser.add_argument('-hiddendim', type=int, dest='hiddendim', help='hidden dimension',default=1500)
    parser.add_argument('-outputdim', type=int, dest='outputdim', help='output dimension')
    parser.add_argument('-bs', type=int, dest='bs', help='batch size', default=512)
    parser.add_argument('-optim', type=str, dest='optim', help='optimizer choice', default='Adam')
    parser.add_argument('-lr', type=float, dest='lr', help='learning rate', default=0.001)
    parser.add_argument('-decay', type=float, dest='decay', help='weight decay', default=0)
    parser.add_argument('-mod', type=str, dest='mod', help='mod', default='ear') # ear crm

    A = parser.parse_args()
    print('Arguments loaded!')
    if A.mod == 'ear':
        #inputdim = 89
        #inputdim = 8787
        inputdim = len(cfg.tag_map)*2+cfg.MAX_TURN+8
        with open('../../data/FM-train-data/tag_question_map.json', 'r') as f:
            outputdim = len(json.load(f)) + 1
    else:
        inputdim = 33
    print("hi: ", inputdim)
    PN = PolicyNetwork(input_dim=inputdim, dim1=A.hiddendim, output_dim=outputdim)
    print(f"input_dim: {inputdim}\thidden_dim: {A.hiddendim}\toutput_dim: {outputdim}")
    cuda_(PN)
    print('Model on GPU')
    data_list = list()

    #dir = '../../data/pretrain-numpy-data-{}'.format(A.mod)
    dir = '../../data/RL-pretrain-data-{}'.format(A.mod)

    files = os.listdir(dir)
    file_paths = [dir + '/' + f for f in files]

    i = 0
    for fp in file_paths:
        with open(fp, 'rb') as f:
            try:
                data_list += pickle.load(f)
                i += 1
            except:
                pass
    print('total files: {}'.format(i))
    data_list = data_list[: int(len(data_list) / 1.5)]
    print('length of data list is: {}'.format(len(data_list)))

    random.shuffle(data_list)

    train_list = data_list
    valid_list = []
    test_list = []
    # train_list = data_list[: int(len(data_list) * 0.7)]
    # valid_list = data_list[int(len(data_list) * 0.7): int(len(data_list) * 0.9)]
    #
    # test_list = data_list[int(len(data_list) * 0.9):]
    # print('train length: {}, valid length: {}, test length: {}'.format(len(train_list), len(valid_list), len(test_list)))
    # sleep(1)  # let you see this

    if A.optim == 'Ada':
        optimizer = torch.optim.Adagrad(PN.parameters(), lr=A.lr, weight_decay=A.decay)
    if A.optim == 'Adam':
        optimizer = torch.optim.Adam(PN.parameters(), lr=A.lr, weight_decay=A.decay)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(8):
        print(f"epoch:{epoch}")
        random.shuffle(train_list)
        model_name = '../../data/PN-model-{}/pretrain-model.pt'.format(A.mod)
        train(A.bs, train_list, valid_list, test_list, optimizer, PN, criterion, epoch, model_name)

MAX_EPOCH = 20
if __name__ == '__main__':
    main()