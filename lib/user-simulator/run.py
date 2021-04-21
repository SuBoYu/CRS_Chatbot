# BB-8 and R2-D2 are best friends.

import sys

sys.path.insert(0, '../FM')
sys.path.insert(0, '../yelp')

import pickle
import torch
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import argparse

import time
import numpy as np

from config import global_config as cfg
from epi import run_one_episode, update_PN_model
from pn import PolicyNetwork
import copy
import random
import json

from collections import defaultdict

the_max = 0
for k, v in cfg.item_dict.items():
    if the_max < max(v['feature_index']):
        the_max = max(v['feature_index'])
print('The max is: {}'.format(the_max))
FEATURE_COUNT = the_max + 1


def cuda_(var):
    return var.cuda() if torch.cuda.is_available() else var


def main(epoch):
    parser = argparse.ArgumentParser(description="Run conversational recommendation.")
    parser.add_argument('-mt', type=int, dest='mt', help='MAX_TURN', default=15)
    parser.add_argument('-playby', type=str, dest='playby', help='playby', default='policy')
    # options include:
    # AO: (Ask Only and recommend by probability)
    # RO: (Recommend Only)
    # policy: (action decided by our policy network)
    parser.add_argument('-fmCommand', type=str, dest='fmCommand', help='fmCommand', default=8)
    # the command used for FM, check out /EAR/lastfm/FM/
    parser.add_argument('-optim', type=str, dest='optim', help='optimizer', default='SGD')
    # the optimizer for policy network
    parser.add_argument('-lr', type=float, dest='lr', help='lr', default=0.001)
    # learning rate of policy network
    parser.add_argument('-decay', type=float, dest='decay', help='decay', default=0)
    # weight decay
    parser.add_argument('-TopKTaxo', type=int, dest='TopKTaxo', help='TopKTaxo', default=3)
    # how many 2-layer feature will represent a big feature. Only Yelp dataset use this param, lastFM have no effect.
    parser.add_argument('-gamma', type=float, dest='gamma', help='gamma', default=0.7)
    # gamma of training policy network
    parser.add_argument('-trick', type=int, dest='trick', help='trick', default=0)
    # whether use normalization in training policy network
    parser.add_argument('-startFrom', type=int, dest='startFrom', help='startFrom', default=0)  # 85817
    # startFrom which user-item interaction pair
    parser.add_argument('-endAt', type=int, dest='endAt', help='endAt', default=20000)
    # endAt which user-item interaction pair
    parser.add_argument('-strategy', type=str, dest='strategy', help='strategy', default='maxent')
    # strategy to choose question to ask, only have effect
    parser.add_argument('-eval', type=int, dest='eval', help='eval', default=0)
    # whether current run is for evaluation
    parser.add_argument('-mini', type=int, dest='mini', help='mini', default=0)
    # means `mini`-batch update the FM
    parser.add_argument('-alwaysupdate', type=int, dest='alwaysupdate', help='alwaysupdate', default=0)
    # means always mini-batch update the FM, alternative is that only do the update for 1 time in a session.
    # we leave this exploration tof follower of our work.
    parser.add_argument('-initeval', type=int, dest='initeval', help='initeval', default=0)
    # whether do the evaluation for the `init`ial version of policy network (directly after pre-train,default=)
    parser.add_argument('-upoptim', type=str, dest='upoptim', help='upoptim', default='SGD')
    # optimizer for reflection stafe
    parser.add_argument('-upcount', type=int, dest='upcount', help='upcount', default=0)
    # how many times to do reflection
    parser.add_argument('-upreg', type=float, dest='upreg', help='upreg', default=0.001)
    # regularization term in
    parser.add_argument('-code', type=str, dest='code', help='code', default='stable')
    # We use it to give each run a unique identifier.
    parser.add_argument('-purpose', type=str, dest='purpose', help='purpose', default='train')
    # options: pretrain, others
    parser.add_argument('-mod', type=str, dest='mod', help='mod', default='ear')
    # options: CRM, EAR
    parser.add_argument('-mask', type=int, dest='mask', help='mask', default=0)
    # use for ablation study, 1, 2, 3, 4 represent our four segments, {ent, sim, his, len}

    A = parser.parse_args()

    cfg.change_param(playby=A.playby, eval=A.eval, update_count=A.upcount, update_reg=A.upreg, purpose=A.purpose,
                     mod=A.mod, mask=A.mask)

    random.seed(1)

    # we random shuffle and split the valid and test set, for Action Stage training and evaluation respectively, to avoid the bias in the dataset.
    all_list = cfg.valid_list + cfg.test_list
    print('The length of all list is: {}'.format(len(all_list)))
    random.shuffle(all_list)
    the_valid_list = all_list[: int(len(all_list) / 2.0)]
    the_test_list = all_list[int(len(all_list) / 2.0):]
    # the_valid_list = cfg.valid_list
    # the_test_list = cfg.test_list

    gamma = A.gamma
    FM_model = cfg.FM_model

    if A.mod == 'ear':
        # fp = '../../data/PN-model-crm/PN-model-crm.txt'
        if epoch == 0 and cfg.eval == 0:
            fp = '../../data/PN-model-ear/pretrain-model.pt'
        else:
            fp = '../../data/PN-model-ear/model-epoch0'
    INPUT_DIM = 0
    if A.mod == 'ear':
        INPUT_DIM = len(cfg.tag_map)*2+15+8
    if A.mod == 'crm':
        INPUT_DIM = 4382
    PN_model = PolicyNetwork(input_dim=INPUT_DIM, dim1=1500, output_dim=len(cfg.tag_map)+1)
    start = time.time()

    try:
        print('fp is: {}'.format(fp))
        PN_model.load_state_dict(torch.load(fp))
        print('load PN model success. ')
    except:
        print('Cannot load the model!!!!!!!!!\nfp is: {}'.format(fp))
        # if A.playby == 'policy':
        #     sys.exit()

    if A.optim == 'Adam':
        optimizer = torch.optim.Adam(PN_model.parameters(), lr=A.lr, weight_decay=A.decay)
    if A.optim == 'SGD':
        optimizer = torch.optim.SGD(PN_model.parameters(), lr=A.lr, weight_decay=A.decay)
    if A.optim == 'RMS':
        optimizer = torch.optim.RMSprop(PN_model.parameters(), lr=A.lr, weight_decay=A.decay)

    numpy_list = list()
    NUMPY_COUNT = 0

    sample_dict = defaultdict(list)
    conversation_length_list = list()
    # endAt = len(the_valid_list) if cfg.eval == 0 else len(the_test_list)
    endAt = len(the_valid_list)
    print(f'endAt: {endAt}')
    print('-'*10)
    print('Train mode' if cfg.eval == 0 else 'Test mode')
    print('-' * 10)
    for epi_count in range(A.startFrom, endAt):
        if epi_count % 1 == 0:
            print('-----\nEpoch: {}\tIt has processed {}/{} episodes'.format(epoch, epi_count, endAt))
        start = time.time()
        cfg.actionProb = epi_count/endAt


        # if A.test == 1 or A.eval == 1:
        if A.eval == 1:
            # u, item = the_test_list[epi_count]
            u, item = the_valid_list[epi_count]
        else:
            u, item = the_valid_list[epi_count]

        if A.purpose == 'fmdata':
            u, item = 0, epi_count

        if A.purpose == 'pretrain':
            u, item = cfg.train_list[epi_count]

        current_FM_model = copy.deepcopy(FM_model)
        param1, param2 = list(), list()
        param3 = list()
        param4 = list()
        i = 0
        for name, param in current_FM_model.named_parameters():
            param4.append(param)
            # print(name, param)
            if i == 0:
                param1.append(param)
            else:
                param2.append(param)
            if i == 2:
                param3.append(param)
            i += 1
        optimizer1_fm = torch.optim.Adagrad(param1, lr=0.01, weight_decay=A.decay)
        optimizer2_fm = torch.optim.SGD(param4, lr=0.001, weight_decay=A.decay)

        user_id = int(u)
        item_id = int(item)

        write_fp = '../../data/interaction-log/{}/v4-code-{}-s-{}-e-{}-lr-{}-gamma-{}-playby-{}-stra-{}-topK-{}-trick-{}-eval-{}-init-{}-mini-{}-always-{}-upcount-{}-upreg-{}-m-{}.txt'.format(
            A.mod.lower(), A.code, A.startFrom, A.endAt, A.lr, A.gamma, A.playby, A.strategy, A.TopKTaxo, A.trick,
            A.eval, A.initeval,
            A.mini, A.alwaysupdate, A.upcount, A.upreg, A.mask)

        choose_pool = cfg.item_dict[str(item_id)]['categories']

        if A.purpose not in ['pretrain', 'fmdata']:
            # this means that: we are not collecting data for pretraining or fm data
            # then we only randomly choose one start attribute to ask!
            choose_pool = [random.choice(choose_pool)]
        # for item_id in cfg.item_dict:
        #     choose_pool = [k for k in cfg.item_dict_rel[str(item_id)] if len(cfg.item_dict_rel[str(item_id)][k]) != 0]
        #     if choose_pool == None:
        #         print(item_id)
        print(f'user id: {user_id}\titem id: {item_id}')
        # choose_pool = [k for k in cfg.item_dict_rel[str(item_id)] if len(cfg.item_dict_rel[str(item_id)][k]) != 0]
        # choose_pool = random.choice(choose_pool)
        for c in choose_pool:
            with open(write_fp, 'a+') as f:
                f.write(
                    'Starting new\nuser ID: {}, item ID: {} episode count: {}, feature: {}\n'.format(user_id, item_id,
                                                                                                     epi_count,
                                                                                                     cfg.item_dict[
                                                                                                         str(item_id)][
                                                                                                         'categories']))
            start_facet = c
            if A.purpose != 'pretrain':
                log_prob_list, rewards, hl = run_one_episode(current_FM_model, user_id, item_id, A.mt, False, write_fp,
                                                         A.strategy, A.TopKTaxo,
                                                         PN_model, gamma, A.trick, A.mini,
                                                         optimizer1_fm, optimizer2_fm, A.alwaysupdate, start_facet,
                                                         A.mask, sample_dict)
                if cfg.eval == 0:
                    with open(f'../../data/train_rec{epoch}.txt', 'a+') as f:
                        # f.writelines(str(rewards.tolist()))
                        f.writelines(str(hl))
                        f.writelines('\n')
                else:
                    with open('../../data/test_rec.txt', 'a+') as f:
                        # f.writelines(str(rewards.tolist()))
                        f.writelines(str(hl))
                        f.writelines('\n')
            else:
                current_np = run_one_episode(current_FM_model, user_id, item_id, A.mt, False, write_fp,
                                             A.strategy, A.TopKTaxo,
                                             PN_model, gamma, A.trick, A.mini,
                                             optimizer1_fm, optimizer2_fm, A.alwaysupdate, start_facet,
                                             A.mask, sample_dict)
                numpy_list += current_np

            # update PN model
            if A.playby == 'policy' and A.eval != 1:
                update_PN_model(PN_model, log_prob_list, rewards, optimizer)
                print('updated PN model')
                current_length = len(log_prob_list)
                conversation_length_list.append(current_length)
            # end update

            if A.purpose != 'pretrain':
                with open(write_fp, 'a') as f:
                    f.write('Big features are: {}\n'.format(choose_pool))
                    if rewards is not None:
                        f.write('reward is: {}\n'.format(rewards.data.numpy().tolist()))
                    f.write('WHOLE PROCESS TAKES: {} SECONDS\n'.format(time.time() - start))

        # Write to pretrain numpy.
        if A.purpose == 'pretrain':
            if len(numpy_list) > 5000:
                with open('../../data/pretrain-numpy-data-{}/segment-{}-start-{}-end-{}.pk'.format(
                        A.mod, NUMPY_COUNT, A.startFrom, A.endAt), 'wb') as f:
                    pickle.dump(numpy_list, f)
                    print('Have written 5000 numpy arrays!')
                NUMPY_COUNT += 1
                numpy_list = list()
        # numpy_list is a list of list.
        # e.g. numpy_list[0][0]: int, indicating the action.
        # numpy_list[0][1]: a one-d array of length 89 for EAR, and 33 for CRM.
        # end write

        # Write sample dict:
        if A.purpose == 'fmdata' and A.playby != 'AOO_valid':
            if epi_count % 100 == 1:
                with open('../../data/sample-dict/start-{}-end-{}.json'.format(A.startFrom, A.endAt), 'w') as f:
                    json.dump(sample_dict, f, indent=4)
        # end write
        if A.purpose == 'fmdata' and A.playby == 'AOO_valid':
            if epi_count % 100 == 1:
                with open('../../data/sample-dict/valid-start-{}-end-{}.json'.format(A.startFrom, A.endAt),
                          'w') as f:
                    json.dump(sample_dict, f, indent=4)

        check_span = 500
        if epi_count % check_span == 0 and epi_count >= 3 * check_span and cfg.eval != 1 and A.purpose != 'pretrain':
            # We use AT (average turn of conversation) as our stopping criterion
            # in training mode, save RL model periodically
            # save model first
            # PATH = '../../data/PN-model-{}/v4-code-{}-s-{}-e-{}-lr-{}-gamma-{}-playby-{}-stra-{}-topK-{}-trick-{}-eval-{}-init-{}-mini-{}-always-{}-upcount-{}-upreg-{}-m-{}.txt'.format(
            #     A.mod.lower(), A.code, A.startFrom, A.endAt, A.lr, A.gamma, A.playby, A.strategy, A.TopKTaxo, A.trick,
            #     A.eval, A.initeval,
            #     A.mini, A.alwaysupdate, A.upcount, A.upreg, A.mask, epi_count)
            PATH = f'../../data/PN-model-{A.mod}/model-epoch{epoch}'
            torch.save(PN_model.state_dict(), PATH)
            print('Model saved at {}'.format(PATH))

            # a0 = conversation_length_list[epi_count - 4 * check_span: epi_count - 3 * check_span]
            a1 = conversation_length_list[epi_count - 3 * check_span: epi_count - 2 * check_span]
            a2 = conversation_length_list[epi_count - 2 * check_span: epi_count - 1 * check_span]
            a3 = conversation_length_list[epi_count - 1 * check_span:]
            a1 = np.mean(np.array(a1))
            a2 = np.mean(np.array(a2))
            a3 = np.mean(np.array(a3))

            with open(write_fp, 'a') as f:
                f.write('$$$current turn: {}, a3: {}, a2: {}, a1: {}\n'.format(epi_count, a3, a2, a1))
            print('current turn: {}, a3: {}, a2: {}, a1: {}'.format(epi_count, a3, a2, a1))

            num_interval = int(epi_count / check_span)
            for i in range(num_interval):
                ave = np.mean(np.array(conversation_length_list[i * check_span: (i + 1) * check_span]))
                print('start: {}, end: {}, average: {}'.format(i * check_span, (i + 1) * check_span, ave))
                # PATH = '../../data/PN-model-{}/v4-code-{}-s-{}-e-{}-lr-{}-gamma-{}-playby-{}-stra-{}-topK-{}-trick-{}-eval-{}-init-{}-mini-{}-always-{}-upcount-{}-upreg-{}-m-{}.txt'.format(
                #     A.mod.lower(), A.code, A.startFrom, A.endAt, A.lr, A.gamma, A.playby, A.strategy, A.TopKTaxo,
                #     A.trick,
                #     A.eval, A.initeval,
                #     A.mini, A.alwaysupdate, A.upcount, A.upreg, A.mask, (i + 1) * check_span)
                PATH = f'../../data/PN-model-{A.mod}/model-epoch{epoch}'
                print('Model saved at: {}'.format(PATH))

            if a3 > a1 and a3 > a2:
                print('Early stop of RL!')
                if cfg.eval == 1:
                    exit()
                else:
                    return


if __name__ == '__main__':
    prob = np.arange(0.4, 1.1, 0.1)
    for i, actProb in enumerate(prob):
        cfg.actionProb = actProb
        main(i)
