import env_demo
import agent
from config import global_config as cfg
from message import message
import random
import torch
from torch.autograd import Variable
import numpy as np


def choose_start_facet(busi_id):
    choose_pool = list()
    if cfg.item_dict[str(busi_id)]['stars'] is not None:
        choose_pool.append('stars')
    if cfg.item_dict[str(busi_id)]['city'] is not None:
        choose_pool.append('city')
    if cfg.item_dict[str(busi_id)]['RestaurantsPriceRange2'] is not None:
        choose_pool.append('RestaurantsPriceRange2')

    THE_FEATURE = random.choice(choose_pool)

    return THE_FEATURE


def get_reward(history_list, gamma, trick, data=None):
    prev_reward = - 0.01

    # -2: reach maximum turn, end.
    # -1: recommend unsuccessful
    # 0: ask attribute, unsuccessful
    # 1: ask attribute, successful
    # 2: recommend successful!

    # r_dict = {
    #     2: 1 + prev_reward,
    #     1: 0.1 + prev_reward,
    #     0: 0 + prev_reward,
    #     -1: 0 + prev_reward,
    #     -2: -0.3
    # }
    r_dict = {
        2: 4,
        1: 1,
        0: -0.5,
        -1: -0.1,
        -2: -2
    }

    # reward_list = [r_dict[item] for item in history_list]
    reward_list = []
    for item in history_list:
        if item != 2:
            reward_list.append(r_dict[item])
        else:
            print('rank: {}/{}'.format(data['ranking'], data['total']))
            reward_list.append(r_dict[2] * (data['total'] - data['ranking'] + 1) / data['total'])
    print('gamma: {}'.format(gamma))

    rewards = []
    R = 0
    for r in reward_list[::-1]:
        R = r + gamma * R
        rewards.insert(0, R)

    # turn rewards to pytorch tensor and standardize
    rewards = torch.Tensor(rewards)
    print('history list: {}'.format(history_list))
    print('reward: {}'.format(rewards))

    # It is a trick for optimization of policy gradient, we can consider use it or not
    # We didn't use it. But the follower of our work can consider use it.
    if trick == 1:
        rewards = (rewards - rewards.mean()) / (rewards.std() + np.finfo(np.float32).eps)

    return rewards


def run_one_episode(FM_model, user_id, busi_id, MAX_TURN, do_random, write_fp, strategy, TopKTaxo, PN_model, gamma, trick, mini, optimizer1_fm, optimizer2_fm,
                    alwaysupdate, start_facet, mask, sample_dict):
    # _______ initialize user and agent _______

    # Initialize the user
    the_user = env_demo.user(user_id, busi_id)
    # Initialize done

    numpy_list = list()
    log_prob_list, reward_list = Variable(torch.Tensor()), list()
    action_tracker, candidate_length_tracker = list(), list()

    the_agent = agent.agent(FM_model, user_id, busi_id, do_random, write_fp, strategy, TopKTaxo,
                            numpy_list, PN_model, log_prob_list, action_tracker,
                            candidate_length_tracker, mini, optimizer1_fm, optimizer2_fm, alwaysupdate, sample_dict)

    # _______ chat history _______
    chat_history = dict()

    # _______ initialize start message _______
    data = dict()
    # data['facet'] = choose_start_facet(busi_id)
    data['facet'] = start_facet
    # print('Starting facet is : {}'.format(data['facet']))
    start_signal = message(cfg.AGENT, cfg.USER, cfg.EPISODE_START, data)

    agent_utterance = None
    while (the_agent.turn_count < MAX_TURN):
        if the_agent.turn_count == 0:
            user_utterance = the_user.response(start_signal)
        else:
            user_utterance = the_user.response(agent_utterance)
        # print('The user utterance in #{} turn, type: {}, data: {}\n'.format(the_agent.turn_count, user_utterance.message_type, user_utterance.data))
        # with open(write_fp, 'a') as f:
        #     f.write('The user utterance in #{} turn, type: {}, data: {}\n'.format(the_agent.turn_count,
        #                                                                           user_utterance.message_type,
        #                                                                           user_utterance.data))

        if user_utterance.message_type == cfg.ACCEPT_REC:
            the_agent.history_list.append(2)
            print('Rec Success! in Turn: {}.'.format(the_agent.turn_count))

            rewards = get_reward(the_agent.history_list, gamma, trick, user_utterance.data)
            return (the_agent.log_prob_list, rewards, the_agent.history_list)

        agent_utterance = the_agent.response(user_utterance)

        the_agent.turn_count += 1

        if the_agent.turn_count == MAX_TURN:
            the_agent.history_list.append(-2)
            print('Max turn quit...')
            rewards = get_reward(the_agent.history_list, gamma, trick)
            return (the_agent.log_prob_list, rewards, the_agent.history_list)


def update_PN_model(model, log_prob_list, rewards, optimizer):
    model.train()

    loss = torch.sum(torch.mul(log_prob_list, Variable(rewards)).mul(-1))
    print(f'loss = {loss}')
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss
# end def
