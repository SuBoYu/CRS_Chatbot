# BB-8 and R2-D2 are best friends.

import sys
sys.path.insert(0, '../FM')
import json
import pickle
import time
import torch
from FM_old import FactorizationMachine

def cuda_(var):
    return var.cuda() if torch.cuda.is_available() else var


class _Config():
    def __init__(self):
        self.init_basic()
        self.init_type()

        self.init_misc()
        self.init_test()
        self.init_FM_related()
        self.actionProb = 0.5

    def init_basic(self):
        dir = 'FM-train-data'

        # with open('../../data/{}/review_dict_train.json'.format(dir), 'r') as f:
        #     self._train_user_to_items = json.load(f)
        # with open('../../data/{}/review_dict_valid.json'.format(dir), 'r') as f:
        #     self._valid_user_to_items = json.load(f)
        # with open('../../data/{}/review_dict_test.json'.format(dir), 'r') as f:
        #     self._test_user_to_items = json.load(f)
        with open('../../data/{}/FM_busi_list.pickle'.format(dir), 'rb') as f:
            self.busi_list = pickle.load(f)
        with open('../../data/{}/FM_user_list.pickle'.format(dir), 'rb') as f:
            self.user_list = pickle.load(f)
        with open('../../data/{}/FM_train_list.pickle'.format(dir), 'rb') as f:
            self.train_list = pickle.load(f)
        with open('../../data/{}/FM_valid_list.pickle'.format(dir), 'rb') as f:
            self.valid_list = pickle.load(f)
        with open('../../data/{}/FM_test_list.pickle'.format(dir), 'rb') as f:
            self.test_list = pickle.load(f)

        # _______ String to Int _______
        with open('../../data/{}/item_map.json'.format(dir), 'r') as f:
            self.item_map = json.load(f)
        with open('../../data/{}/user_map.json'.format(dir), 'r') as f:
            self.user_map = json.load(f)
        with open('../../data/{}/tag_map.json'.format(dir), 'r') as f:
            self.tag_map = json.load(f)

        self.tag_map_inverted = dict()
        for k, v in self.tag_map.items():
            self.tag_map_inverted[v] = k

        # _______ item info _______
        with open('../../data/{}/item_dict.json'.format(dir), 'r') as f:
            self.item_dict = json.load(f)
            print(len(self.item_dict))
        # with open('../../data/{}/item_dict_rel.json'.format(dir), 'r') as f:
        #     self.item_dict_rel = json.load(f)

    def init_type(self):
        self.INFORM_FACET = 'INFORM_FACET'
        self.ACCEPT_REC = 'ACCEPT_REC'
        self.REJECT_REC = 'REJECT_REC'

        # define agent behavior
        self.ASK_FACET = 'ASK_FACET'
        self.MAKE_REC = 'MAKE_REC'
        self.FINISH_REC_ACP = 'FINISH_REC_ACP'
        self.FINISH_REC_REJ = 'FINISH_REC_REJ'
        self.EPISODE_START = 'EPISODE_START'

        # define the sender type
        self.USER = 'USER'
        self.AGENT = 'AGENT'

    def init_misc(self):
        self.FACET_POOL = list(self.tag_map.keys())
        print('Total feature length is: {}, Top 30 namely: {}'.format(len(self.FACET_POOL), self.FACET_POOL[: 30]))
        self.REC_NUM = 10
        self.MAX_TURN = 5
        self.play_by = None
        self.calculate_all = None

    def init_FM_related(self):
        # category_max = 0
        feature_max = 0
        for k, v in self.item_dict.items():
            # if max(v['categories']) > category_max:
            #     category_max = max(v['categories'])
            if max(v['feature_index']) > feature_max:
                feature_max = max(v['feature_index'])

        self.feature_count = feature_max + 1

        self.PN_model = None
        self.FM_model = None

        # fp = '../../data/FM-model-merge/FM-model-command-8-pretrain-2.pt'
        fp = '../../data/FM-model-merge/v4-FM-lr-0.01-flr-0.001-reg-0.002-decay-0-qonly-1-bs-64-command-6-hs-64-ip-0.01-dr-0.5-optim-Ada-oldnew-new-pretrain-0-uf-0-rd-0-freeze-0-seed-330-useremb-1epoch-442.pt'

        model = FactorizationMachine(emb_size=64, user_length=len(self.user_list), item_length=len(self.item_dict),
                                     feature_length=feature_max + 1, qonly=1, command=8, hs=64, ip=0.01,
                                     dr=0.5, old_new='new')


        start = time.time()
        model.load_state_dict(torch.load(fp))
        print('load FM model {} takes: {} secs'.format(fp, time.time() - start))
        self.emb_matrix = model.feature_emb.weight[..., :-1].detach().numpy()
        self.user_emb = model.ui_emb.weight[..., :-1].detach().numpy()
        self.FM_model = cuda_(model)

    def init_test(self):
        pass

    def change_param(self, playby, eval, update_count, update_reg, purpose, mod, mask):
        self.play_by = playby
        self.eval = eval
        self.update_count = update_count
        self.update_reg = update_reg
        self.purpose = purpose
        self.mod = mod
        self.mask = mask

        if self.mod == 'crm' or self.play_by == 'RO':
            category_max = 0
            feature_max = 0
            for k, v in self.item_dict.items():
                if max(v['categories']) > category_max:
                    category_max = max(v['categories'])
                if max(v['feature_index']) > feature_max:
                    feature_max = max(v['feature_index'])
            # fp = '../../data/FM-model-merge/FM-model-command-6-pretrain-0.pt'
            fp = '../../data/FM-model-merge/v4-FM-lr-0.01-flr-0.001-reg-0.002-decay-0-qonly-1-bs-64-command-6-hs-64-ip-0.01-dr-0.5-optim-Ada-oldnew-new-pretrain-0-uf-0-rd-0-freeze-0-seed-330-useremb-1epoch-442.pt'
            model = FactorizationMachine(emb_size=64, user_length=len(self.user_list), item_length=len(self.item_dict),
                                         feature_length=feature_max + 1, qonly=1, command=6, hs=64, ip=0.01,
                                         dr=0.5, old_new='new')
            start = time.time()
            model.load_state_dict(torch.load(fp))
            print('load FM model {} takes: {} secs'.format(fp,time.time() - start))
            self.FM_model = cuda_(model)

start = time.time()
global_config = _Config()
print('Config takes: {}'.format(time.time() - start))

print('___Config Done!!___')
