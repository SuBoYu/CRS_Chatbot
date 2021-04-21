import torch
import json
import pickle
import numpy as np
from sklearn.metrics import roc_auc_score
import time


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

print(train_list[: 10])

busi_list_numpy = np.array(busi_list)
user_list_numpy = np.array(user_list)

the_max = 0
for k, v in item_dict.items():
    if the_max < max(v['feature_index']):
        the_max = max(v['feature_index'])
print('the max is: {}'.format(the_max))
FEATURE_COUNT = the_max + 1
PAD_IDX1 = len(user_list) + len(busi_list)
PAD_IDX2 = FEATURE_COUNT


def predict_feature(model, given_preference, to_test):
    gp = model.feature_emb(torch.LongTensor(given_preference))[..., :-1].detach().numpy()
    emb_weight = model.feature_emb.weight[..., :-1].detach().numpy()
    result = list()

    for test_feature in to_test:
        temp = 0
        for i in range(gp.shape[0]):
            temp += np.inner(gp[i], emb_weight[test_feature])
        result.append(temp)

    return result

def topk(y_true, pred, k):
    y_true_ = y_true[:k]
    pred_ = pred[:k]
    if sum(y_true_) == 0:
        return 0  # TODO: I change it to 0
    else:
        try:
            a = roc_auc_score(y_true_, pred_)
        except:
            a = 0
        return a


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

        full_feature = item_dict[str(item_p_output)]['feature_index']
        preference_feature = preference_list
        residual_preference = list(set(full_feature) - set(preference_feature))
        residual_feature_all = list(set(list(range(FEATURE_COUNT - 1))) - set(full_feature))

        if len(residual_preference) == 0:
            continue
        to_test = residual_feature_all + residual_preference

        predictions = predict_feature(model, preference_feature, to_test)
        predictions = np.array(predictions)


        predictions = predictions.reshape((len(to_test), 1)[0])
        y_true = [0] * len(predictions)
        for i in range(len(residual_preference)):
            y_true[-(i + 1)] = 1
        tmp = list(zip(y_true, predictions))
        tmp = sorted(tmp, key=lambda x: x[1], reverse=True)
        y_true, predictions = zip(*tmp)

        icon = []

        for index, item in enumerate(y_true):
            if item > 0:
                icon.append(index)

        auc = roc_auc_score(y_true, predictions)

        result_list.append((auc, topk(y_true, predictions, 10), topk(y_true, predictions, 50)
                            , topk(y_true, predictions, 100), topk(y_true, predictions, 200),
                            topk(y_true, predictions, 500), len(predictions)))
        i += 1
    return result_list


def evaluate_8(model, epoch, filename):
    model.eval()
    model.cpu()
    tt = time.time()
    pickle_file_path = '../../data/{}/v1-speed-valid-{}.pickle'.format(dir1, 1)
    with open(pickle_file_path, 'rb') as f:
        pickle_file = pickle.load(f)
    print('Open evaluation pickle file: {} takes {} seconds, evaluation length: {}'.format(pickle_file_path, time.time() - tt, len(pickle_file[0])))
    pickle_file_length = len(pickle_file[0])

    start = time.time()
    print('Starting {} epoch'.format(epoch))
    bs = 64
    max_iter = int(pickle_file_length / float(bs))

    max_iter = 100

    result = list()
    for iter_ in range(max_iter):
        if iter_ > 1 and iter_ % 20 == 0:
            print('--')
            print('Takes {} seconds to finish {}% of this task'.format(str(time.time() - start),
                                                                        float(iter_) * 100 / max_iter))
        result += rank_by_batch(pickle_file, iter_, bs, pickle_file_length, model)

    auc_mean = np.mean(np.array([item[0] for item in result]))
    auc_median = np.median(np.array([item[0] for item in result]))
    print('auc mean: {}'.format(auc_mean), 'auc median: {}'.format(auc_median))

    PATH = '../../data/FM-log-merge/' + filename + '.txt'
    with open(PATH, 'a') as f:
        f.write('validating {} epoch on feature prediction\n'.format(epoch))
        auc_mean = np.mean(np.array([item[0] for item in result]))
        auc_median = np.median(np.array([item[0] for item in result]))
        f.write('auc mean: {}\n'.format(auc_mean))
        f.write('auc median: {}\n'.format(auc_median))
        f.flush()
    model.train()
    cuda_(model)

