from collections import Counter
import numpy as np
from config import global_config as cfg
import time


class cal_ent():
    '''
    given the current candidate list, calculate the entropy of every feature(denote by f)
    '''

    def __init__(self, recent_candidate_list):
        self.recent_candidate_list = recent_candidate_list

    def calculate_entropy_for_one_tag(self, tagID, _counter):
        '''
        Args:
        tagID: int
        '''
        v = _counter[tagID]
        p1 = float(v) / len(self.recent_candidate_list)
        p2 = 1.0 - p1

        if p1 < 0.00001 or 1 - p1 < 0.00001:
            return 0
        return (- p1 * np.log2(p1) - p2 * np.log2(p2))

    def do_job(self):
        entropy_dict_small_feature = dict()
        cat_list_all = list()
        for k in self.recent_candidate_list:
            cat_list_all += cfg.item_dict[str(k)]['categories']

        c = Counter(cat_list_all)
        #print('c is: {} (doing entropy calculation)'.format(len(self.recent_candidate_list)))
        for k, v in c.items():
            node_entropy_self = self.calculate_entropy_for_one_tag(k, c)
            # entropy_dict[cfg.tag_map_inverted[k]] = node_entropy_self
            entropy_dict_small_feature[k] = node_entropy_self

        output_dict = dict()
        for key in cfg.FACET_POOL:
            if key not in entropy_dict_small_feature:
                entropy_dict_small_feature[key] = 0
                output_dict[key] = 0
            else:
                output_dict[key] = entropy_dict_small_feature[key]
        # for key in cfg.FACET_POOL:
        #     if int(key) not in entropy_dict_small_feature:
        #         entropy_dict_small_feature[int(key)] = 0
        #         output_dict[int(key)] = 0
        #     else:
        #         output_dict[int(key)] = entropy_dict_small_feature[int(key)]

        return output_dict
        # As we only have so called 'small features' here, directly return
        #return entropy_dict_small_feature
