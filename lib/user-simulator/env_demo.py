# BB-8 and R2-D2 are best friends.

import sys
sys.path.insert(0, '../FM')
sys.path.insert(0, '../yelp')

from collections import Counter
import numpy as np
from random import randint
import json
import random

from message import message
from config import global_config as cfg
import time



class user():
    def __init__(self, user_id, busi_id):
        self.user_id = user_id
        self.busi_id = busi_id
        self.recent_candidate_list = [int(k) for k, v in cfg.item_dict.items()]
        # self.not_told = cfg.item_dict_rel[str(busi_id)].copy()

    def find_brother(self, node):
        return [child.name for child in node.parent.children if child.name != node.name]

    def find_children(self, node):
        return [child.name for child in node.children if child.name != node.name]

    def inform_facet(self, facet, response):
        data = dict()
        data['facet'] = facet

        # N, n
        if response == "n" or response == "N":
            data['value'] = None

        # Y, y
        else:
            data['value'] = [facet]

        return message(cfg.USER, cfg.AGENT, cfg.INFORM_FACET, data)
        # if facet not in cfg.item_dict[str(self.busi_id)]['categories']:
        #     data['value'] = None
        #     return message(cfg.USER, cfg.AGENT, cfg.INFORM_FACET, data)
        # else:
        #     data['value'] = [facet]
        #     return message(cfg.USER, cfg.AGENT, cfg.INFORM_FACET, data)
        # if len(self.not_told[str(facet)]) == 0:
        #     data['value'] = None
        # else:
        #     # val = np.random.randint(0, len(self.not_told[str(facet)]))
        #     # val = self.not_told[str(facet)][val]
        #     # self.not_told[str(facet)] = list(set(self.not_told[str(facet)]) - {val})
        #     # data['value'] = [val]
        #     data['value'] = self.not_told[str(facet)]
        # return message(cfg.USER, cfg.AGENT, cfg.INFORM_FACET, data)

    def response(self, input_message, user_response):
        assert input_message.sender == cfg.AGENT
        assert input_message.receiver == cfg.USER

        # _______ update candidate _______
        # if 'candidate' in input_message.data: self.recent_candidate_list = input_message.data['candidate']

        new_message = None
        if input_message.message_type == cfg.EPISODE_START or input_message.message_type == cfg.ASK_FACET:
            facet = input_message.data['facet']
            # print("input facet: ")
            # facet = int(input())
            new_message = self.inform_facet(facet, user_response)

        if input_message.message_type == cfg.MAKE_REC:

            #推薦時的system output
            # print("recommand list: \n")
            # print(input_message.data['rec_list'])
            # print("hit?(Y/n)")
            # 推薦時的system output

            # 推薦時的user input
            #hit = input()
            # 推薦時的user input

            # if self.busi_id in input_message.data['rec_list']:

            # r, R
            if user_response == "r" or user_response == 'R':
                data = dict()
                data['rejected_item_list'] = input_message.data['rec_list']
                new_message = message(cfg.USER, cfg.AGENT, cfg.REJECT_REC, data)
            # h, H
            elif user_response == "h" or user_response == 'H':
                data = dict()
                data['ranking'] = input_message.data['rec_list'].index(self.busi_id) + 1
                data['total'] = len(input_message.data['rec_list'])
                new_message = message(cfg.USER, cfg.AGENT, cfg.ACCEPT_REC, data)

        return new_message
    # end def
