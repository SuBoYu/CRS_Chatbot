# BB-8 and R2-D2 are best friends.

import sys

sys.path.insert(0, '../FM')
sys.path.insert(0, '../yelp')

import torch
import argparse

from config import global_config as cfg
from epi_demo import run_one_episode
from pn import PolicyNetwork
import copy
import random
import env_demo
import agent
from torch.autograd import Variable
from env_demo import user
from collections import defaultdict
from message import message
import numpy as np
import json


with open('../../data/FM-train-data/item_map.json', 'r') as f:
    r_item_map = {v: k for k, v in json.load(f).items()}

with open('../../data/raw_data/small_metadata.json', 'r') as f:
    meta_data = json.load(f)

the_max = 0
for k, v in cfg.item_dict.items():
    if the_max < max(v['feature_index']):
        the_max = max(v['feature_index'])
print('The max is: {}'.format(the_max))
FEATURE_COUNT = the_max + 1


def cuda_(var):
    return var.cuda() if torch.cuda.is_available() else var


class chat_model:
    def __init__(self, user_name, item_name):
        self.mt = 5
        self.playby = "policy"
        # options include:
        # AO: (Ask Only and recommend by probability)
        # RO: (Recommend Only)
        # policy: (action decided by our policy network)
        self.fmCommand = 8
        # the command used for FM, check out /EAR/lastfm/FM/
        self.optim = "SGD"
        # the optimizer for policy network
        self.lr = 0.001
        # learning rate of policy network
        self.decay = 0
        # weight decay
        self.TopKTaxo = 3
        # how many 2-layer feature will represent a big feature. Only Yelp dataset use this param, lastFM have no effect.
        self.gamma = 0.7
        # gamma of training policy network
        self.trick = 0
        # whether use normalization in training policy network
        self.strategy = "maxent"
        # strategy to choose question to ask, only have effect
        self.eval = 1
        # whether current run is for evaluation
        self.mini = 0
        # means `mini`-batch update the FM
        self.alwaysupdate = 0
        # means always mini-batch update the FM, alternative is that only do the update for 1 time in a session.
        # we leave this exploration tof follower of our work.
        self.initeval = 0
        # whether do the evaluation for the `init`ial version of policy network (directly after pre-train,default=)
        self.upoptim = "SGD"
        # optimizer for reflection stafe
        self.upcount = 0
        # how many times to do reflection
        self.upreg = 0.001
        # regularization term in
        self.code = "stable"
        # We use it to give each run a unique identifier.
        self.purpose = "train"
        # options: pretrain, others
        self.mod = "ear"
        # options: CRM, EAR
        self.mask = 0
        # use for ablation study, 1, 2, 3, 4 represent our four segments, {ent, sim, his, len}

        cfg.change_param(playby=self.playby, eval=self.eval, update_count=self.upcount, update_reg=self.upreg,
                         purpose=self.purpose,
                         mod=self.mod, mask=self.mask)

        gamma = self.gamma
        FM_model = cfg.FM_model

        if self.mod == 'ear':
            fp = '../../data/PN-model-ear/model-epoch0'
        if self.mod == 'ear':
            INPUT_DIM = len(cfg.tag_map)*2+self.mt+8
        self.PN_model = PolicyNetwork(input_dim=INPUT_DIM, dim1=1500, output_dim=len(cfg.tag_map)+1)

        try:
            print('fp is: {}'.format(fp))
            self.PN_model.load_state_dict(torch.load(fp))
            print('load PN model success.')
        except:
            print('Cannot load the model!!!!!!!!!\n fp is: {}'.format(fp))
            sys.exit()

        if self.optim == 'Adam':
            optimizer = torch.optim.Adam(self.PN_model.parameters(), lr=self.lr, weight_decay=self.decay)
        if self.optim == 'SGD':
            optimizer = torch.optim.SGD(self.PN_model.parameters(), lr=self.lr, weight_decay=self.decay)
        if self.optim == 'RMS':
            optimizer = torch.optim.RMSprop(self.PN_model.parameters(), lr=self.lr, weight_decay=self.decay)

        self.sample_dict = defaultdict(list)
        self.conversation_length_list = list()
        # print('-'*10)
        # print('Train mode' if cfg.eval == 0 else 'Test mode')
        # print('-' * 10)

        # cfg.actionProb = epi_count/endAt
        # if A.test == 1 or A.eval == 1:

        # input
        self.u = user_name
        self.item = item_name

        self.current_FM_model = copy.deepcopy(FM_model)
        param1, param2 = list(), list()
        param3 = list()
        param4 = list()
        i = 0
        for name, param in self.current_FM_model.named_parameters():
            param4.append(param)
            # print(name, param)
            if i == 0:
                param1.append(param)
            else:
                param2.append(param)
            if i == 2:
                param3.append(param)
            i += 1
        self.optimizer1_fm = torch.optim.Adagrad(param1, lr=0.01, weight_decay=self.decay)
        self.optimizer2_fm = torch.optim.SGD(param4, lr=0.001, weight_decay=self.decay)

        self.user_id = int(self.u)
        self.item_id = int(self.item)

        # Initialize the user
        self.the_user = user(self.user_id, self.item_id)

        self.numpy_list = list()
        self.log_prob_list, self.reward_list = Variable(torch.Tensor()), list()
        self.action_tracker, self.candidate_length_tracker = list(), list()

        self.the_agent = agent.agent(self.current_FM_model, self.user_id, self.item_id, False, "", self.strategy,
                                     self.TopKTaxo, self.numpy_list, self.PN_model, self.log_prob_list,
                                     self.action_tracker, self.candidate_length_tracker, self.mini, self.optimizer1_fm,
                                     self.optimizer2_fm, self.alwaysupdate, self.sample_dict)

        self.agent_utterance = None

        choose_pool = cfg.item_dict[str(self.item_id)]['categories']
        c = random.choice(choose_pool)
        # print(f'user id: {user_id}\titem id: {item_id}')

        self.start_facet = c

    def first_conversation(self, user_response): # Yes/No/Hit/Reject
        data = dict()
        data['facet'] = self.start_facet
        start_signal = message(cfg.AGENT, cfg.USER, cfg.EPISODE_START, data)
        user_utterance = self.the_user.response(start_signal, user_response)

        # user_utterance.message_type: cfg.REJECT_REC, cfg.ACCEPT_REC, cfg.INFORM_FACET

        # if user_utterance.message_type == cfg.ACCEPT_REC:
        #     self.the_agent.history_list.append(2)
        #     s = 'Rec Success! in Turn:' + str(self.the_agent.turn_count) + '.'
        print(user_utterance.message_type)
        self.agent_utterance = self.the_agent.response(user_utterance)

        print(self.agent_utterance.message_type)

        # agent_utterance.message_type: cfg.ASK_FACET, cfg.MAKE_REC

        if self.agent_utterance.message_type == cfg.ASK_FACET:
            s = "D" + str(self.agent_utterance.data['facet'])

        elif self.agent_utterance.message_type == cfg.MAKE_REC:
            s = []
            s.append("r")
            k = []
            for i, j in enumerate(self.agent_utterance.data["rec_list"][:5]):
                # j = meta_data[r_item_map[int(j)]]["title"]
                j = str(j)
                #j = j.split(" ")
                # if type(j) == list:
                #     j = j[-3]+" "+j[-2]+" "+j[-1]
                # # if i != len(self.agent_utterance.data["rec_list"])-1:
                # if i != 4:
                #     s += str(i+1)+". "+str(j) + "\n"
                # else:
                #     s += str(i+1)+". "+str(j)
                k.append(j)
            s.append(k)

        self.the_agent.turn_count += 1

        return s

    def conversation(self, user_response):  # Yes/No/Hit/Reject
        if self.the_agent.turn_count < self.mt:
            user_utterance = self.the_user.response(self.agent_utterance, user_response)

            if user_utterance.message_type == cfg.ACCEPT_REC:
                self.the_agent.history_list.append(2)
                s = 'Rec Success! in Turn:' + str(self.the_agent.turn_count) + '.'
                return s

            self.agent_utterance = self.the_agent.response(user_utterance)

            if self.agent_utterance.message_type == cfg.ASK_FACET:
                s = "D" + str(self.agent_utterance.data['facet'])

            elif self.agent_utterance.message_type == cfg.MAKE_REC:
                s = []
                s.append("r")
                k = []
                for i, j in enumerate(self.agent_utterance.data["rec_list"][:5]):
                    # j = meta_data[r_item_map[int(j)]]["title"]
                    j = str(j)
                    # j = j.split(" ")
                    # if type(j) == list:
                    #     j = j[-3]+" "+j[-2]+" "+j[-1]
                    # # if i != len(self.agent_utterance.data["rec_list"])-1:
                    # if i != 4:
                    #     s += str(i+1)+". "+str(j) + "\n"
                    # else:
                    #     s += str(i+1)+". "+str(j)
                    k.append(j)
                s.append(k)

            self.the_agent.turn_count += 1

        if self.the_agent.turn_count == self.mt: # 改成>的話，mt才是5，但要等Model 那邊重train，agent那邊max_turn < 5-2也要改
            self.the_agent.history_list.append(-2)
            s = "Already meet the max turn of conversation: " + str(self.mt)

        return s
