# BB-8 and R2-D2 are best friends.

import sys
import time
from collections import defaultdict
import random
random.seed(0)
sys.path.insert(0, '../FM')
sys.path.insert(0, '../yelp')

import json
import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_value_, clip_grad_norm_
from torch.distributions import Categorical


from message import message
from config import global_config as cfg
from utils_entropy import cal_ent
from heapq import nlargest, nsmallest
from utils_fea_sim import feature_similarity
from utils_sense import rank_items
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_sequence

import math


def cuda_(var):
    return var.cuda() if torch.cuda.is_available() else var
# end def


class agent():
    def __init__(self, FM_model, user_id, busi_id, do_random, write_fp, strategy, TopKTaxo, numpy_list, PN_model, log_prob_list, action_tracker, candidate_length_tracker, mini, optimizer1_fm, optimizer2_fm, alwaysupdate, sample_dict):
        #_______ input parameters_______
        self.user_id = user_id
        self.busi_id = busi_id
        self.FM_model = FM_model

        self.turn_count = 0
        self.F_dict = defaultdict(lambda: defaultdict())
        self.recent_candidate_list = [int(k) for k, v in cfg.item_dict.items()]
        self.recent_candidate_list_ranked = self.recent_candidate_list

        self.asked_feature = list()
        self.do_random = do_random
        self.rejected_item_list_ = list()

        self.history_list = list()

        self.write_fp = write_fp
        self.strategy = strategy
        self.TopKTaxo = TopKTaxo
        self.entropy_dict_10 = None
        self.entropy_dict_50 = None
        self.entropy_dict = None
        self.sim_dict = None
        self.sim_dict2 = None
        self.PN_model = PN_model

        self.known_feature = list()

        self.residual_feature_big = None
        self.change = None
        self.skip_big_feature = list()
        self.numpy_list = numpy_list

        self.log_prob_list = log_prob_list
        self.action_tracker = action_tracker
        self.candidate_length_tracker = candidate_length_tracker
        self.mini_update_already = False
        self.mini = mini
        self.optimizer1_fm = optimizer1_fm
        self.optimizer2_fm = optimizer2_fm
        self.alwaysupdate = alwaysupdate
        self.update_this_turn = False
        self.feature_length = 4382
        self.big_feature_length = 4382
        self.sample_dict = sample_dict
        # self.ask_rec = [len(cfg.item_dict_rel[str(busi_id)][rel]) for rel in cfg.item_dict_rel[str(busi_id)]]
        # self.ask_rel = []
        self.asked_feature = []
    # end def

    def get_batch_data(self, pos_neg_pairs, bs, iter_):
        # this function is used for Reflection Stage.
        # Get batched data for updating FM model
        PAD_IDX1 = len(cfg.user_list) + len(cfg.item_dict)
        PAD_IDX2 = cfg.feature_count

        left = iter_ * bs
        right = min((iter_ + 1) * bs, len(pos_neg_pairs))
        pos_list, pos_list2, neg_list, neg_list2 = list(), list(), list(), list()
        for instance in pos_neg_pairs[left: right]:
            #instance[0]: pos item, instance[1] neg item
            pos_list.append(torch.LongTensor([self.user_id, instance[0] + len(cfg.user_list)]))
            neg_list.append(torch.LongTensor([self.user_id, instance[1] + len(cfg.user_list)]))
        # end for
        preference_list = torch.LongTensor(self.known_feature).expand(len(pos_list), len(self.known_feature))

        pos_list = pad_sequence(pos_list, batch_first=True, padding_value=PAD_IDX1)
        pos_list2 = preference_list

        neg_list = pad_sequence(neg_list, batch_first=True, padding_value=PAD_IDX1)
        neg_list2 = preference_list

        return cuda_(pos_list), cuda_(pos_list2), cuda_(neg_list), cuda_(neg_list2)
    # end def

    def mini_update_FM(self):
        # This function is used to update the pretrained FM model.
        self.FM_model.train()
        bs = 32
        pos_items = list(set(cfg._train_user_to_items[str(self.user_id)]) - set([self.busi_id]))
        neg_items = self.rejected_item_list_[-10:]

        # to remove all ground truth interacted items ...
        random_neg = list(set(range(len(cfg.item_dict))) - set(cfg._train_user_to_items[str(self.user_id)])
                          - set(cfg._valid_user_to_items[str(self.user_id)])
                          - set(cfg._test_user_to_items[str(self.user_id)]))

        pos_items = pos_items + random.sample(random_neg, len(pos_items))  # add some random negative samples to avoid overfitting
        neg_items = neg_items + random.sample(random_neg, len(neg_items))  # add some random negative samples to avoid overfitting

        #_______ Form Pair _______
        pos_neg_pairs = list()
        for p_item in pos_items:
            for n_item in neg_items:
                pos_neg_pairs.append((p_item, n_item))

        pos_neg_pairs = list()

        num = int(bs / len(pos_items)) + 1
        pos_items = pos_items * num

        for p_item in pos_items:
            n_item = random.choice(neg_items)
            pos_neg_pairs.append((p_item, n_item))
        random.shuffle(pos_neg_pairs)

        max_iter = int(len(pos_neg_pairs) / bs)

        reg_ = torch.Tensor([cfg.update_reg])
        reg_ = torch.autograd.Variable(reg_, requires_grad=False)
        reg_ = cuda_(reg_)
        reg = reg_

        lsigmoid = nn.LogSigmoid()
        for iter_ in range(max_iter):
            pos_list, pos_list2, neg_list, neg_list2 = self.get_batch_data(pos_neg_pairs, bs, iter_)
            result_pos, feature_bias_matrix_pos, nonzero_matrix_pos = self.FM_model(pos_list, None, pos_list2)
            result_neg, feature_bias_matrix_neg, nonzero_matrix_neg = self.FM_model(neg_list, None, neg_list2)
            diff = (result_pos - result_neg)
            loss = - lsigmoid(diff).sum(dim=0)

            nonzero_matrix_pos_ = (nonzero_matrix_pos ** 2).sum(dim=2).sum(dim=1, keepdim=True)
            nonzero_matrix_neg_ = (nonzero_matrix_neg ** 2).sum(dim=2).sum(dim=1, keepdim=True)
            loss += (reg * nonzero_matrix_pos_).sum(dim=0)
            loss += (reg * nonzero_matrix_neg_).sum(dim=0)

            self.optimizer2_fm.zero_grad()
            loss.backward()
            self.optimizer2_fm.step()
        # end for
    # end def


    def vectorize_crm(self):
        a = [0] * self.feature_length
        # print(max(self.known_feature))
        for item in self.known_feature:
            a[item] = 1
        return np.array(a)
    # end def

    def vectorize(self):
        # Following line: it means the entropy of first 10 items. We didn't use it, but follower of our work can try.
        list1 = [v for k, v in self.entropy_dict_10.items()]
        list2 = [v for k, v in self.entropy_dict_50.items()]

        list3 = [v for k, v in self.entropy_dict.items()]
        list4 = [v for k, v in self.sim_dict2.items()]

        assert len(list3) == len(cfg.FACET_POOL)
        assert len(list4) == len(cfg.FACET_POOL)

        MAX_TURN = 5
        list5 = self.history_list + [0] * (MAX_TURN - len(self.history_list))

        list6 = [0] * 8
        if len(self.recent_candidate_list) <= 10:
            list6[0] = 1
        if len(self.recent_candidate_list) > 10 and len(self.recent_candidate_list) <= 50:
            list6[1] = 1
        if len(self.recent_candidate_list) > 50 and len(self.recent_candidate_list) <= 100:
            list6[2] = 1
        if len(self.recent_candidate_list) > 100 and len(self.recent_candidate_list) <= 200:
            list6[3] = 1
        if len(self.recent_candidate_list) > 200 and len(self.recent_candidate_list) <= 300:
            list6[4] = 1
        if len(self.recent_candidate_list) > 300 and len(self.recent_candidate_list) <= 500:
            list6[5] = 1
        if len(self.recent_candidate_list) > 500 and len(self.recent_candidate_list) <= 1000:
            list6[6] = 1
        if len(self.recent_candidate_list) > 1000:
            list6[7] = 1

        if cfg.mask == 1:
            list3 = [0] * len(list3)
        if cfg.mask == 2:
            list4 = [0] * len(list4)
        if cfg.mask == 3:
            list5 = [0] * len(list5)
        if cfg.mask == 4:
            list6 = [0] * len(list6)

        list_cat = list3 + list4 + list5 + list6
        list_cat = np.array(list_cat)
        #assert len(list_cat) == 89
        return list_cat
    # end def

    def update_upon_feature_inform(self, input_message):
        assert input_message.message_type == cfg.INFORM_FACET

        #_______ update F_dict________
        facet = input_message.data['facet']
        self.asked_feature.append(str(facet))
        value = input_message.data['value']

        if value is not None:
            self.update_this_turn = True
            rcl = []
            for k in self.recent_candidate_list:
                if set(value).issubset(set(cfg.item_dict[str(k)]['categories'])):
                    rcl.append(k)
            # self.recent_candidate_list = [k for k in self.recent_candidate_list if set(value).issubset(set(cfg.item_dict[str(k)]['categories']))]
            self.recent_candidate_list = rcl
            self.recent_candidate_list = list(set(self.recent_candidate_list) - set([self.busi_id])) + [self.busi_id]
            self.known_feature.append(cfg.tag_map[str(value[0])])
            self.known_feature = list(set(self.known_feature))

            # following dict
            l = list(set(self.recent_candidate_list) - set([self.busi_id]))
            random.shuffle(l)
            if cfg.play_by == 'AOO':
                self.sample_dict[self.busi_id].append((self.known_feature, l[: 10]))
            if cfg.play_by == 'AOO_valid':
                self.sample_dict[self.busi_id].append((self.known_feature, l[: 300]))
            # end dictionary

        if cfg.play_by != 'AOO' and cfg.play_by != 'AOO_valid':
            self.sim_dict = feature_similarity(self.known_feature, self.user_id, self.TopKTaxo)
            self.sim_dict2 = self.sim_dict.copy()

            self.recent_candidate_list = list(set(self.recent_candidate_list) - set([self.busi_id])) + [self.busi_id]
            self.recent_candidate_list_ranked = rank_items(self.known_feature, self.user_id, self.busi_id,self.skip_big_feature, self.FM_model, self.recent_candidate_list, self.write_fp, 0)

        if value is not None and value[0] is not None:
            c = cal_ent(self.recent_candidate_list)
            d = c.do_job()
            self.entropy_dict = d

            c = cal_ent(self.recent_candidate_list[: 10])
            d = c.do_job()
            self.entropy_dict_10 = d

            c = cal_ent(self.recent_candidate_list[: 50])
            d = c.do_job()
            self.entropy_dict_50 = d

        for f in self.asked_feature:
            # self.entropy_dict[int(f)] = 0
            self.entropy_dict[f] = 0

        if cfg.play_by == 'AOO' or cfg.play_by == 'AOO_valid':
            return

        for f in self.asked_feature:
            if self.sim_dict is not None and str(f) in self.sim_dict:
                self.sim_dict[str(f)] = -1

        for f in self.asked_feature:
            if self.sim_dict2 is not None and str(f) in self.sim_dict:
                self.sim_dict2[str(f)] = -1

        residual_feature = cfg.item_dict[str(self.busi_id)]['categories']
        # known_ = [int(cfg.tag_map_inverted[item]) for item in self.known_feature]
        known_ = [cfg.tag_map_inverted[item] for item in self.known_feature]
        residual_feature = list(set(residual_feature) - set(known_))
        self.residual_feature_big = residual_feature
    # end def

    def prepare_next_question(self):
        if self.strategy == 'maxent':
            facet = max(self.entropy_dict, key=self.entropy_dict.get)
            data = dict()
            data['facet'] = facet
            new_message = message(cfg.AGENT, cfg.USER, cfg.ASK_FACET, data)
            self.asked_feature.append(facet)
            return new_message
        # end if max ent

        elif self.strategy == 'maxsim':
            # ask following max similarity between attributes
            for f in self.asked_feature:
                if self.sim_dict is not None and f in self.sim_dict:
                    self.sim_dict[f] = -1
            if len(self.known_feature) == 0 or self.sim_dict is None:
               facet = max(self.entropy_dict, key=self.entropy_dict.get)
            else:
               facet = max(self.sim_dict, key=self.sim_dict.get)
            data = dict()
            data['facet'] = int(facet)
            new_message = message(cfg.AGENT, cfg.USER, cfg.ASK_FACET, data)
            self.asked_feature.append(facet)
            return new_message
        # end if maxsim
        else:
            pool = [item for item in cfg.FACET_POOL if item not in self.asked_feature]
            facet = np.random.choice(np.array(pool), 1)[0]
            data = dict()
            if facet in [item.name for item in cfg.cat_tree.children]:
                data['facet'] = facet
            else:
                data['facet'] = facet
            new_message = message(cfg.AGENT, cfg.USER, cfg.ASK_FACET, data)
            return new_message
        # end if others
    # end def

    def prepare_rec_message(self):
        self.recent_candidate_list_ranked = [item for item in self.recent_candidate_list_ranked if item not in self.rejected_item_list_]  # Delete those has been rejected
        rec_list = self.recent_candidate_list_ranked[: 10]
        data = dict()
        data['rec_list'] = rec_list
        new_message = message(cfg.AGENT, cfg.USER, cfg.MAKE_REC, data)
        return new_message
    # end def

    def response(self, input_message):
        '''
        The agent moves a step forward, upon receiving a message from the user.
        '''
        assert input_message.sender == cfg.USER
        assert input_message.receiver == cfg.AGENT

        self.update_this_turn = False

        #_______ update the agent self_______
        if input_message.message_type == cfg.INFORM_FACET:
            self.update_upon_feature_inform(input_message)
        if input_message.message_type == cfg.REJECT_REC:
            self.rejected_item_list_ += input_message.data['rejected_item_list']
            if self.mini == 1:
                if self.alwaysupdate == 1 or self.mini_update_already is False:
                    for i in range(cfg.update_count):
                        self.mini_update_FM()
                    self.mini_update_already = True
                    self.recent_candidate_list = list(set(self.recent_candidate_list) - set(self.rejected_item_list_))
                    self.recent_candidate_list = list(set(self.recent_candidate_list) - set([self.busi_id])) + [self.busi_id]
                    self.recent_candidate_list_ranked = rank_items(self.known_feature, self.user_id, self.busi_id, self.skip_big_feature, self.FM_model, self.recent_candidate_list, self.write_fp, 1)

        #_______ Adding into history _______
        if input_message.message_type == cfg.INFORM_FACET:
            if self.turn_count > 0:  # means first doesn't count
                if input_message.data['value'] is None:
                    self.history_list.append(0)  # ask attribute, fail
                else:
                    self.history_list.append(1)  # ask attribute, successful
        if input_message.message_type == cfg.REJECT_REC:
            self.history_list.append(-1)  # try recommendation, user doesn't want.
            self.recent_candidate_list = list(set(self.recent_candidate_list) - set(self.rejected_item_list_))  # don't consider

        if cfg.play_by != 'AOO' and cfg.play_by != 'AOO_valid':
            # Add control point here
            if cfg.mod == 'ear':
                state_vector = self.vectorize()
            else:
                state_vector = self.vectorize_crm()

        action = None
        SoftMax = nn.Softmax(dim=-1)
        if cfg.play_by == 'AOO' or cfg.play_by == 'AOO_valid':
            new_message = self.prepare_next_question()

        if cfg.play_by == 'AO':  # means Ask, the recommendation is made by a probability
            new_message = self.prepare_next_question()
            x = len(self.recent_candidate_list)
            p = 10.0 / x
            a = random.uniform(0, 1)
            if a < p:
                new_message = self.prepare_rec_message()

        if cfg.play_by == 'RO':
            # means RecOnly, only make recommendation at each turn.
            # For Abs-Greedy Evaluation
            new_message = self.prepare_rec_message()

        if cfg.play_by == 'policy':
            s = torch.from_numpy(state_vector).float()
            s = Variable(s, requires_grad=False)
            self.PN_model.eval()
            # print(torch.min(s))
            pred = self.PN_model(s)
            # print(pred.detach().numpy())
            for feat in self.asked_feature:
                pred[cfg.tag_map[feat]] = -math.inf

            prob = SoftMax(pred)
            # print(prob.detach().numpy())
            # print(torch.min(prob))
            c = Categorical(prob)

            # use max prob
            if cfg.eval == 1 or cfg.eval == 0:
                pred_data = pred.data.tolist()
                sorted_index = sorted(range(len(pred_data)), key=lambda k: pred_data[k], reverse=True)
                print('Top 5 action: {}'.format(sorted_index[:5]))
                print('Value of top 5 actions: {}'.format([pred_data[v] for v in sorted_index[:5]]))
                print('Ranking of recommendation action: {}'.format(sorted_index.index(len(cfg.tag_map))))
                print('Value of recommendation action: {}'.format(pred_data[len(cfg.tag_map)]))

                unasked_max = None
                # for item in sorted_index:
                #     if item <= 7:
                #         unasked_max = item
                #         break
                if self.turn_count < 5 - 2:
                    unasked_max = sorted_index[0]
                else:
                    unasked_max = pred.shape[0] - 1
                action = Variable(torch.tensor([unasked_max], dtype=torch.long))  # make it compatible with torch
            else:
                # for training of Action stage
                i = 0
                action_ = pred.shape[0]
                ap = np.random.random()
                if ap >= cfg.actionProb:
                # if ap > 1:
                    print('sample action')
                    # while i < 10000:
                    #     action_ = c.sample()
                    #     i += 1
                    #     if action_ <= 7:
                    #         break
                    # action = action_
                    action = c.sample()
                    action = Variable(torch.tensor([action], dtype=torch.long))
                else:
                    print('max prob action')
                    pred_data = pred.data.tolist()
                    sorted_index = sorted(range(len(pred_data)), key=lambda k: pred_data[k], reverse=True)
                    unasked_max = sorted_index[0]
                    action = Variable(torch.tensor([unasked_max], dtype=torch.long))
                    # action = Variable(torch.IntTensor([unasked_max]))  # make it compatible with torch
            print('action is: {}'.format(action.numpy()[0]))

            log_prob = c.log_prob(action)
            if self.turn_count != 0:
                self.log_prob_list = torch.cat([self.log_prob_list, log_prob.reshape(1)])
            else:
                self.log_prob_list = log_prob.reshape(1)

            if action < pred.shape[0]-1:
                # self.ask_rel.append(action)
                data = dict()
                # data['facet'] = int(cfg.FACET_POOL[action])
                data['facet'] = cfg.FACET_POOL[action]
                new_message = message(cfg.AGENT, cfg.USER, cfg.ASK_FACET, data)
            else:
                new_message = self.prepare_rec_message()
        # end if policy

        if cfg.play_by == 'policy':
            self.action_tracker.append(action.data.numpy().tolist())
            self.candidate_length_tracker.append(len(self.recent_candidate_list))

        # following are for writing to numpy array
        # action = None
        # if new_message.message_type == cfg.ASK_FACET:
        #     action = cfg.FACET_POOL.index(str(new_message.data['facet']))

        # if new_message.message_type == cfg.MAKE_REC:
        #     action = len(cfg.FACET_POOL)
        #
        # if cfg.purpose != 'fmdata':
        #     self.numpy_list.append((action, state_vector))
        # # end following
        #
        # inverted_known_feature = [int(cfg.tag_map_inverted[item]) for item in self.known_feature]
        # self.residual_feature_big = list(set(cfg.item_dict[str(self.busi_id)]['categories']) - set(inverted_known_feature))
        #
        # with open(self.write_fp, 'a') as f:
        #     f.write('Turn count: {}, candidate length: {}\n'.format(self.turn_count, len(self.recent_candidate_list)))

        return new_message
    # end def
