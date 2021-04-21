import random
import torch
import json
import pickle
import numpy as np
from sklearn.metrics import roc_auc_score
import time
from torch.nn.utils.rnn import pad_sequence


def cuda_(var):
    return var.cuda() if torch.cuda.is_available() else var

dir = 'FM-train-data-lastfm'
dir1 = 'FM-sample-data-lastfm'

with open('../../data/{}/review_dict_train.json'.format(dir), 'r') as f:
    _train_user_to_items = json.load(f)
with open('../../data/{}/review_dict_valid.json'.format(dir), 'r') as f:
    _valid_user_to_items = json.load(f)
with open('../../data/{}/review_dict_test.json'.format(dir), 'r') as f:
    _test_user_to_items = json.load(f)


def load_data():
    with open('../../data/{}/FM_busi_list.pickle'.format(dir), 'rb') as f:
        busi_list = pickle.load(f)

    with open('../../data/{}/FM_user_list.pickle'.format(dir), 'rb') as f:
        user_list = pickle.load(f)

    with open('../../data/{}/FM_train_list.pickle'.format(dir), 'rb') as f:
        train_list = pickle.load(f)

    with open('../../data/{}/FM_valid_list.pickle'.format(dir), 'rb') as f:
        valid_list = pickle.load(f)

    with open('../../data/{}/FM_test_list.pickle'.format(dir), 'rb') as f:
        test_list = pickle.load(f)

    with open('../../data/{}/item_dict.json'.format(dir), 'r') as f:
        item_dict = json.load(f)

    return busi_list, user_list, train_list, valid_list, test_list, item_dict

busi_list, user_list, train_list, valid_list, test_list, item_dict = load_data()

print('train_list length is: {}'.format(len(train_list)))
print('busi_list length is: {}'.format(len(busi_list)))

print('train list top 10: {}'.format(train_list[: 10]))

busi_list_numpy = np.array(busi_list)
user_list_numpy = np.array(user_list)

the_max = 0
for k, v in item_dict.items():
    if the_max < max(v['feature_index']):
        the_max = max(v['feature_index'])
print('evaluate the max is: {}'.format(the_max))
FEATURE_COUNT = the_max + 1
PAD_IDX1 = len(user_list) + len(busi_list)
PAD_IDX2 = FEATURE_COUNT


def topk(y_true, pred, k):
    y_true_ = y_true[:k]
    pred_ = pred[:k]
    if sum(y_true_) == 0:
        return 0
    else:
        return roc_auc_score(y_true_, pred_)


def rank_by_batch(pickle_file, iter_, bs, pickle_file_length, model):
    '''
    user_output, item_p_output, i_neg2_output, preference_list = list(), list(), list(), list()
    '''
    left, right = iter_ * bs, min(pickle_file_length, (iter_ + 1) * bs)

    I = pickle_file[0][left:right]
    II = pickle_file[1][left:right]
    III = pickle_file[2][left:right]
    IV = pickle_file[3][left:right]

    i = 0
    index_none = list()

    for user_output, item_p_output, i_neg2_output, preference_list in zip(I, II, III, IV):
        if i_neg2_output is None or len(i_neg2_output) == 0:
            index_none.append(i)
        i += 1

    i = 0
    result_list = list()
    for user_output, item_p_output, i_neg2_output, preference_list in zip(I, II, III, IV):
        if i in index_none:
            i += 1
            continue

        total_list = list(i_neg2_output)[: 1000] + [item_p_output]

        user_input = [user_output] * len(total_list)

        pos_list, pos_list2= list(), list()
        cumu_length = 0
        for instance in zip(user_input, total_list):
            new_list = list()
            new_list.append(instance[0])
            new_list.append(instance[1] + len(user_list))
            pos_list.append(torch.LongTensor(new_list))
            f = item_dict[str(instance[1])]['feature_index']
            cumu_length += len(f)
            pos_list2.append(torch.LongTensor(f))

        pos_list = pad_sequence(pos_list, batch_first=True, padding_value=PAD_IDX1)
        prefer_list = torch.LongTensor(preference_list).expand(len(total_list), len(preference_list))
        # ADD by hc
        if cumu_length != 0:
            pos_list2.sort(key=lambda x: -1 * x.shape[0])
            pos_list2 = pad_sequence(pos_list2, batch_first=True, padding_value=PAD_IDX2)
        else:
            pos_list2 = torch.LongTensor([PAD_IDX2]).expand(pos_list.shape[0], 1)


        predictions, _, _ = model(cuda_(pos_list), cuda_(pos_list2), cuda_(prefer_list))
        predictions = predictions.detach().cpu().numpy()

        mini_gtitems = [item_p_output]
        num_gt = len(mini_gtitems)
        num_neg = len(total_list) - num_gt
        predictions = predictions.reshape((num_neg + 1, 1)[0])
        y_true = [0] * len(predictions)
        y_true[-1] = 1
        tmp = list(zip(y_true, predictions))
        tmp = sorted(tmp, key=lambda x: x[1], reverse=True)
        y_true, predictions = zip(*tmp)
        auc = roc_auc_score(y_true, predictions)

        result_list.append((auc, topk(y_true, predictions, 10), topk(y_true, predictions, 50)
                            , topk(y_true, predictions, 100), topk(y_true, predictions, 200),
                            topk(y_true, predictions, 500), len(predictions)))
        a = topk(y_true, predictions, 10)

        i += 1
    return result_list


def evaluate_7(model, epoch, filename):
    model.eval()
    tt = time.time()
    pickle_file_path = '../../data/{}/v1-speed-valid-{}.pickle'.format(dir1, 1)
    with open(pickle_file_path, 'rb') as f:
        pickle_file = pickle.load(f)
    print('Open evaluation pickle file: {} takes {} seconds, evaluation length: {}'.format(pickle_file_path, time.time() - tt, len(pickle_file[0])))
    pickle_file_length = len(pickle_file[0])

    print('Starting {} epoch'.format(epoch))
    bs = 64
    max_iter = int(pickle_file_length / float(bs))
    # Only do 20 iteration for the sake of time
    max_iter = 20

    result = list()
    for iter_ in range(max_iter):
        result += rank_by_batch(pickle_file, iter_, bs, pickle_file_length, model)

    auc_mean = np.mean(np.array([item[0] for item in result]))
    auc_median = np.median(np.array([item[0] for item in result]))
    print('auc mean: {}'.format(auc_mean), 'auc median: {}'.format(auc_median))

    PATH = '../../data/FM-log-merge/' + filename + '.txt'
    with open(PATH, 'a') as f:
        f.write('validating {} epoch on item prediction\n'.format(epoch))
        auc_mean = np.mean(np.array([item[0] for item in result]))
        auc_median = np.median(np.array([item[0] for item in result]))
        f.write('auc mean: {}\n'.format(auc_mean))
        f.write('auc median: {}\n'.format(auc_median))
        f.flush()
    model.train()
