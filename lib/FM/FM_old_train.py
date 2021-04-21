import random
import torch
import torch.nn as nn
import json
import pickle
import numpy as np
import time
from torch.nn.utils.rnn import pad_sequence

import argparse

from FM_old import FactorizationMachine
from FM_old_evaluate import evaluate_7
from FM_tune_predict import  evaluate_8


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


global busi_list, user_list, train_list, valid_list, test_list, item_dict
busi_list, user_list, train_list, valid_list, test_list, item_dict = load_data()
#random.shuffle(train_list)

print('train_list length is: {}'.format(len(train_list)))
print('busi_list length is: {}'.format(len(busi_list)))

print('train list top 10: {}'.format(train_list[: 10]))

busi_list_numpy = np.array(busi_list)
user_list_numpy = np.array(user_list)


the_max = 0
for k, v in item_dict.items():
    if the_max < max(v['feature_index']):
        the_max = max(v['feature_index'])
print('The max is: {}'.format(the_max))
FEATURE_COUNT = the_max + 1
PAD_IDX1 = len(user_list) + len(busi_list)
PAD_IDX2 = FEATURE_COUNT



def translate_pickle_to_data(pickle_file, iter_, bs, pickle_file_length, uf):
    '''
    user_pickle = pickle_file[0]
    item_p_pickle = pickle_file[1]
    i_neg1_pickle = pickle_file[2]
    i_neg2_pickle = pickle_file[3]
    preference_pickle = pickle_file[4]
    '''
    left, right = iter_ * bs, min(pickle_file_length, (iter_ + 1) * bs)

    pos_list, pos_list2, neg_list, neg_list2, new_neg_list, new_neg_list2, preference_list_1, preference_list_2 = [], [], [], [], [], [], [], []

    I = pickle_file[0][left:right]
    II = pickle_file[1][left:right]
    III = pickle_file[2][left:right]
    IV = pickle_file[3][left:right]
    V = pickle_file[4][left:right]

    residual_feature, neg_feature = None, None
    if uf == 1:
        feature_range = np.arange(FEATURE_COUNT).tolist()
        residual_feature, neg_feature = [], []
        for user_pickle, item_p_pickle, i_neg1_pickle, i_neg2_pickle, preference_pickle in zip(I, II, III, IV, V):
            gt_feature = item_dict[str(item_p_pickle)]['feature_index']
            this_residual_feature = list(set(item_dict[str(item_p_pickle)]['feature_index']) - set(preference_pickle))
            remain_feature = list(set(feature_range) - set(gt_feature))
            this_neg_feature = np.random.choice(remain_feature, len(this_residual_feature))
            residual_feature.append(torch.LongTensor(this_residual_feature))
            neg_feature.append(torch.LongTensor(this_neg_feature))
        residual_feature = pad_sequence(residual_feature, batch_first=True, padding_value=PAD_IDX2)
        neg_feature = pad_sequence(neg_feature, batch_first=True, padding_value=PAD_IDX2)

    i = 0
    index_none = list()
    for user_pickle, item_p_pickle, i_neg1_pickle, i_neg2_pickle, preference_pickle in zip(I, II, III, IV, V):
        pos_list.append(torch.LongTensor([user_pickle, item_p_pickle + len(user_list)]))
        f = item_dict[str(item_p_pickle)]['feature_index']
        pos_list2.append(torch.LongTensor(f))
        neg_list.append(torch.LongTensor([user_pickle, i_neg1_pickle + len(user_list)]))
        f = item_dict[str(i_neg1_pickle)]['feature_index']
        neg_list2.append(torch.LongTensor(f))

        p = preference_pickle
        preference_list_1.append(torch.LongTensor(p))
        if i_neg2_pickle is None:
            index_none.append(i)
        i += 1

    i = 0
    for user_pickle, item_p_pickle, i_neg1_pickle, i_neg2_pickle, preference_pickle in zip(I, II, III, IV, V):
        if i in index_none:
            i += 1
            continue
        new_neg_list.append(torch.LongTensor([user_pickle, i_neg2_pickle + len(user_list)]))
        f = item_dict[str(i_neg2_pickle)]['feature_index']
        new_neg_list2.append(torch.LongTensor(f))
        preference_list_2.append(torch.LongTensor(preference_pickle))
        i += 1

    pos_list = pad_sequence(pos_list, batch_first=True, padding_value=PAD_IDX1)
    pos_list2 = pad_sequence(pos_list2, batch_first=True, padding_value=PAD_IDX2)
    neg_list = pad_sequence(neg_list, batch_first=True, padding_value=PAD_IDX1)
    neg_list2 = pad_sequence(neg_list2, batch_first=True, padding_value=PAD_IDX2)
    new_neg_list = pad_sequence(new_neg_list, batch_first=True, padding_value=PAD_IDX1)
    new_neg_list2 = pad_sequence(new_neg_list2, batch_first=True, padding_value=PAD_IDX2)
    preference_list_1 = pad_sequence(preference_list_1, batch_first=True, padding_value=PAD_IDX2)
    preference_list_2 = pad_sequence(preference_list_2, batch_first=True, padding_value=PAD_IDX2)

    if uf != 0:
        return cuda_(pos_list), cuda_(pos_list2), cuda_(neg_list), cuda_(neg_list2), cuda_(new_neg_list), cuda_(
            new_neg_list2), cuda_(preference_list_1), cuda_(preference_list_2), index_none, cuda_(residual_feature), cuda_(neg_feature)
    else:
        return cuda_(pos_list), cuda_(pos_list2), cuda_(neg_list), cuda_(neg_list2), cuda_(new_neg_list), cuda_(
            new_neg_list2), cuda_(preference_list_1), cuda_(preference_list_2), index_none, residual_feature, neg_feature


def train(model, bs, max_epoch, optimizer1, optimizer2, optimizer3, reg, qonly, observe, command, filename, uf, use_useremb):
    model.train()
    lsigmoid = nn.LogSigmoid()
    reg_float = float(reg.data.cpu().numpy()[0])

    for epoch in range(max_epoch):
        # _______ Do the evaluation _______
        if epoch % observe == 0 and epoch > 0:
            print('Evaluating on item prediction')
            evaluate_7(model, epoch, filename)
            print('Evaluating on feature similarity')
            evaluate_8(model, epoch, filename)

        tt = time.time()
        pickle_file_path = '../../data/{}/v1-speed-train-{}.pickle'.format(dir1, epoch)
        if uf == 1:
            pickle_file_path = '../../data/{}/v1-speed-train-{}.pickle'.format(dir1, epoch + 30)
        with open(pickle_file_path, 'rb') as f:
            pickle_file = pickle.load(f)
        print('Open pickle file: {} takes {} seconds'.format(pickle_file_path, time.time() - tt))
        pickle_file_length = len(pickle_file[0])

        model.train()

        mix = list(zip(pickle_file[0], pickle_file[1], pickle_file[2], pickle_file[3], pickle_file[4]))
        random.shuffle(mix)
        I, II, III, IV, V = zip(*mix)
        new_pk_file = [I, II, III, IV, V]

        start = time.time()
        print('Starting {} epoch'.format(epoch))
        epoch_loss = 0
        epoch_loss_2 = 0
        max_iter = int(pickle_file_length / float(bs))

        for iter_ in range(max_iter):
            if iter_ > 1 and iter_ % 100 == 0:
                print('--')
                print('Takes {} seconds to finish {}% of this epoch'.format(str(time.time() - start),
                                                                            float(iter_) * 100 / max_iter))
                print('loss is: {}'.format(float(epoch_loss) / (bs * iter_)))
                print('iter_:{} Bias grad norm: {}, Static grad norm: {}, Preference grad norm: {}'.format(iter_,torch.norm(model.Bias.grad),torch.norm(model.ui_emb.weight.grad),torch.norm(model.feature_emb.weight.grad)))

            pos_list, pos_list2, neg_list, neg_list2, new_neg_list, new_neg_list2, preference_list_1, preference_list_new, index_none, residual_feature, neg_feature \
                = translate_pickle_to_data(new_pk_file, iter_, bs, pickle_file_length, uf)

            optimizer1.zero_grad()
            optimizer2.zero_grad()
            result_pos, feature_bias_matrix_pos, nonzero_matrix_pos = model(pos_list, pos_list2,
                                                                            preference_list_1)  # (bs, 1), (bs, 2, 1), (bs, 2, emb_size)

            result_neg, feature_bias_matrix_neg, nonzero_matrix_neg = model(neg_list, neg_list2, preference_list_1)
            diff = (result_pos - result_neg)
            loss = - lsigmoid(diff).sum(dim=0)  # The Minus is crucial is

            if command in [8]:
                # The second type of negative sample
                new_result_neg, new_feature_bias_matrix_neg, new_nonzero_matrix_neg = model(new_neg_list, new_neg_list2, preference_list_new)

                # Reason for this is that, sometimes the sample is missing, so we have to also omit that in result_pos
                T = cuda_(torch.tensor([]))
                for i in range(bs):
                    if i in index_none:
                        continue
                    T = torch.cat([T, result_pos[i]], dim=0)

                T = T.view(T.shape[0], -1)
                assert T.shape[0] == new_result_neg.shape[0]
                diff = T - new_result_neg
                if loss is not None:
                    loss += - lsigmoid(diff).sum(dim=0)
                else:
                    loss = - lsigmoid(diff).sum(dim=0)

            # regularization
            if reg_float != 0:
                if qonly != 1:
                    feature_bias_matrix_pos_ = (feature_bias_matrix_pos ** 2).sum(dim=1)  # (bs, 1)
                    feature_bias_matrix_neg_ = (feature_bias_matrix_neg ** 2).sum(dim=1)  # (bs, 1)
                    nonzero_matrix_pos_ = (nonzero_matrix_pos ** 2).sum(dim=2).sum(dim=1, keepdim=True)  # (bs, 1)
                    nonzero_matrix_neg_ = (nonzero_matrix_neg ** 2).sum(dim=2).sum(dim=1, keepdim=True)  # (bs, 1)
                    new_nonzero_matrix_neg_ = (new_nonzero_matrix_neg_ ** 2).sum(dim=2).sum(dim=1, keepdim=True)
                    regular_norm = (feature_bias_matrix_pos_ + feature_bias_matrix_neg_ + nonzero_matrix_pos_ + nonzero_matrix_neg_ + new_nonzero_matrix_neg_)
                    loss += (reg * regular_norm).sum(dim=0)
                else:
                    nonzero_matrix_pos_ = (nonzero_matrix_pos ** 2).sum(dim=2).sum(dim=1, keepdim=True)
                    nonzero_matrix_neg_ = (nonzero_matrix_neg ** 2).sum(dim=2).sum(dim=1, keepdim=True)
                    loss += (reg * nonzero_matrix_pos_).sum(dim=0)
                    loss += (reg * nonzero_matrix_neg_).sum(dim=0)
            epoch_loss += loss.data
            loss.backward()
            optimizer1.step()
            optimizer2.step()

            if uf == 1:
                # updating feature embedding
                # we try to optimize
                A = model.feature_emb(preference_list_1)[..., :-1]
                user_emb = model.ui_emb(pos_list[:, 0])[..., :-1].unsqueeze(dim=1).detach()
                if use_useremb == 1:
                    A = torch.cat([A, user_emb], dim=1)
                B = model.feature_emb(residual_feature)[..., :-1]
                C = model.feature_emb(neg_feature)[..., :-1]

                D = torch.matmul(A, B.transpose(2, 1))
                E = torch.matmul(A, C.transpose(2, 1))

                p_vs_residual = D.view(D.shape[0], -1, 1)
                p_vs_neg = E.view(E.shape[0], -1, 1)

                p_vs_residual = p_vs_residual.sum(dim=1)
                p_vs_neg = p_vs_neg.sum(dim=1)
                diff = (p_vs_residual - p_vs_neg)
                temp = - lsigmoid(diff).sum(dim=0)
                loss = temp
                epoch_loss_2 += temp.data

                if iter_ % 1000 == 0 and iter_ > 0:
                    print('2ND iter_:{} preference grad norm: {}'.format(iter_, torch.norm(model.feature_emb.weight.grad)))
                    print('2ND loss is: {}'.format(float(epoch_loss_2) / (bs * iter_)))

                optimizer3.zero_grad()
                loss.backward()
                optimizer3.step()

            # These line is to make an alert on console when we meet gradient explosion.
            if iter_ > 0 and iter_ % 1 == 0:
                if torch.norm(model.ui_emb.weight.grad) > 100 or torch.norm(model.feature_emb.weight.grad) > 500:
                    print('iter_:{} Bias grad norm: {}, F-bias grad norm: {}, F-embedding grad norm: {}'.format(iter_,torch.norm(model.Bias.grad),torch.norm(model.ui_emb.weight.grad),torch.norm(model.feature_emb.weight.grad)))

            # Uncomment this to use clip gradient norm (but currently we don't need)
            # clip_grad_norm_(model.ui_emb.weight, 5000)
            # clip_grad_norm_(model.feature_emb.weight, 5000)

        print('epoch loss: {}'.format(epoch_loss / pickle_file_length))
        print('epoch loss 2: {}'.format(epoch_loss_2 / pickle_file_length))

        if epoch % 1 == 0:
            PATH = '../../data/FM-model-merge/' + filename + 'epoch-{}.pt'.format(epoch)
            torch.save(model.state_dict(), PATH)
            print('Model saved at {}'.format(PATH))

        PATH = '../../data/FM-log-merge/' + filename + '.txt'
        with open(PATH, 'a') as f:
            f.write('Starting {} epoch\n'.format(epoch))
            f.write('training loss 1: {}\n'.format(epoch_loss / len(train_list)))
            f.write('training loss 2: {}\n'.format(epoch_loss_2 / len(train_list)))


def main():
    parser = argparse.ArgumentParser(description="Run FM")
    parser.add_argument('-lr', type=float, metavar='<lr>', dest='lr', help='lr', default= 0.01 )
    parser.add_argument('-flr', type=float, metavar='<flr>', dest='flr', help='flr', default=0.001)
    # means the learning rate of feature similarity learning
    parser.add_argument('-reg', type=float, metavar='<reg>', dest='reg', help='reg', default=0.002)
    # regularization
    parser.add_argument('-decay', type=float, metavar='<decay>', dest='decay', help='decay', default=0)
    # weight decay
    parser.add_argument('-qonly', type=int, metavar='<qonly>', dest='qonly', help='qonly', default=1)
    # means quadratic form only (letting go other terms in FM equation...)
    parser.add_argument('-bs', type=int, metavar='<bs>', dest='bs', help='bs', default=64)
    # batch size
    parser.add_argument('-hs', type=int, metavar='<hs>', dest='hs', help='hs', default=64)
    # hidden size
    parser.add_argument('-ip', type=float, metavar='<ip>', dest='ip', help='ip', default=0.01)
    # init parameter for hidden
    parser.add_argument('-dr', type=float, metavar='<dr>', dest='dr', help='dr', default=0.5)
    # dropout ratio
    parser.add_argument('-optim', type=str, metavar='<optim>', dest='optim', help='optim', default='Ada')
    # optimizer
    parser.add_argument('-observe', type=int, metavar='<observe>', dest='observe', help='observe', default=1)
    # the frequency of doing evaluation
    parser.add_argument('-oldnew', type=str, metavar='<oldnew>', dest='oldnew', help='oldnew', default='new')
    # we don't use this parameter now
    parser.add_argument('-pretrain', type=int, metavar='<pretrain>', dest='pretrain', help='pretrain', default=0)
    # does it need to load pretrain model?
    parser.add_argument('-uf', type=int, metavar='<uf>', dest='uf', help='uf', default=0)
    # update feature
    parser.add_argument('-rd', type=int, metavar='<rd>', dest='rd', help='rd', default= 0)
    # remove duplicate, we don;t use this parameter now
    parser.add_argument('-useremb', type=int, metavar='<useremb>', dest='useremb', help='user embedding', default=1)
    # update user embedding during feature similarity
    parser.add_argument('-freeze', type=int, metavar='<freeze>', dest='freeze', help='freeze', default=0)
    # we don't use this param now
    parser.add_argument('-command', type=int, metavar='<command>', dest='command', help='command', default=6)
    # command = 6: normal FM
    # command = 8: with our second type of negative sample
    parser.add_argument('-seed', type=int, metavar='<seed>', dest='seed', help='seed', default=330)
    # random seed
    A = parser.parse_args()

    user_length = len(user_list)
    item_length = len(busi_list)

    random.seed(A.seed)
    np.random.seed(A.seed)
    torch.manual_seed(A.seed)
    torch.cuda.manual_seed(A.seed)

    if A.pretrain == 0:
        # means no pretrain
        model = FactorizationMachine(emb_size=A.hs, user_length=user_length, item_length=item_length,
                                     feature_length=FEATURE_COUNT, qonly=A.qonly, command=A.command, hs=A.hs, ip=A.ip, dr=A.dr, old_new=A.oldnew)

    if A.pretrain == 2:
        model = FactorizationMachine(emb_size=A.hs, user_length=user_length, item_length=item_length,
                                     feature_length=FEATURE_COUNT, qonly=A.qonly, command=A.command, hs=A.hs, ip=A.ip, dr=A.dr, old_new=A.oldnew)
        fp = '../../data/FM-model-merge/v4-FM-lr-0.01-flr-0.001-reg-0.002-decay-0.0-qonly-1-bs-64-command-8-hs-64-ip-0.01-dr-0.5-optim-SGD-oldnew-new-pretrain-0-uf-0-rd-0-freeze-0-seed-3812-useremb-1epoch-49.pt'
        print(fp)
        model.load_state_dict(torch.load(fp))
    cuda_(model)

    param1, param2 = list(), list()
    param3 = list()

    i = 0
    for name, param in model.named_parameters():
        print(name, param)
        if i == 0:
            param1.append(param)
        else:
            param2.append(param)
        if i == 2:
            param3.append(param)
        i += 1


    print('param1 is: {}, shape:{}\nparam2 is: {}, shape: {}\nparam3 is: {}, shape: {}\n'.format(param1, [param.shape for param in param1], param2, [param.shape for param in param2], param3, [param.shape for param in param3]))
    bs = A.bs
    max_epoch = 250

    if A.optim == 'SGD':
        optimizer1 = torch.optim.SGD(param1, lr=A.lr, weight_decay=0.1)
        optimizer2 = torch.optim.SGD(param2, lr=A.lr)
        optimizer3 = torch.optim.SGD(param3, lr=A.flr)
    if A.optim == 'Ada':
        optimizer1 = torch.optim.Adagrad(param1, lr=A.lr, weight_decay=A.decay)
        optimizer2 = torch.optim.Adagrad(param2, lr=A.lr, weight_decay=A.decay)
        optimizer3 = torch.optim.Adagrad(param3, lr=A.flr, weight_decay=A.decay)

    reg_ = torch.Tensor([A.reg])
    reg_ = torch.autograd.Variable(reg_, requires_grad=False)
    reg_ = cuda_(reg_)

    file_name = 'v4-FM-lr-{}-flr-{}-reg-{}-decay-{}-qonly-{}-bs-{}-command-{}-hs-{}-ip-{}-dr-{}-optim-{}-oldnew-{}-pretrain-{}-uf-{}-rd-{}-freeze-{}-seed-{}-useremb-{}'.format(A.lr, A.flr, A.reg, A.decay, A.qonly,
                                                                                     A.bs, A.command, A.hs, A.ip, A.dr, A.optim, A.oldnew, A.pretrain, A.uf, A.rd, A.freeze, A.seed, A.useremb)

    model = train(model, bs, max_epoch, optimizer1, optimizer2, optimizer3, reg_, A.qonly, A.observe, A.command, file_name, A.uf, A.useremb)


if __name__ == '__main__':
    main()
