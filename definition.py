#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2025 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


from kaiwu_agent.utils.common_func import create_cls, attached
import numpy as np
import math

SampleData = create_cls("SampleData", rewards=None, done= None, last_state=None)
ObsData = create_cls("ObsData", feature=None, legal_actions=None, done = None)
ActData = create_cls("ActData", act=None)

@attached
def sample_process(list_game_data, gamma, last_state):
    rewards = []
    dones = []
    discounted_reward = 0
    for sample in list_game_data:
        rewards.append(sample.reward)
        dones.append(sample.done)
        #discounted_reward = sample.reward + (gamma * discounted_reward)
        #rewards.insert(0, discounted_reward)
        
    return SampleData(rewards=rewards, done = dones, last_state= last_state)


def reward_shaping(frame_no, score, terminated, truncated, obs, _obs, extra_info, _extra_info, step):
    reward = 0
    #print(_obs["feature"])
    game_info, next_game_info = extra_info['game_info'], _extra_info['game_info']
    next_pos_x, next_pos_z = next_game_info["pos_x"], next_game_info["pos_z"]
    end_treasure_dists, next_end_treasure_dists = obs["feature"], _obs["feature"]
    next_local_view = next_game_info["local_view"]
    '''
    reward = 0
    # 发现宝箱的奖励
    if 4 in next_local_view:
        reward += 50
    # 发现终点的奖励
    if 3 in next_local_view:
        reward += 80
    # 靠近宝箱的奖励
    treasure_dist, next_treasure_dist = end_treasure_dists[1:], next_end_treasure_dists[1:]
    nearest_treasure_index = np.argmin(treasure_dist)
    if treasure_dist[nearest_treasure_index] > next_treasure_dist[nearest_treasure_index]:
        reward += 25
    else:
        # 远离宝箱的惩罚
        reward -= 25

    # 靠近终点的奖励
    end_dist, next_end_dist = end_treasure_dists[0], next_end_treasure_dists[0]
    if end_dist > next_end_dist:
        reward += 30'''
    total_score = next_game_info["total_score"]
    treasure_score = next_game_info["treasure_score"]
    treasure_count = next_game_info["treasure_count"]
    # 靠近终点的奖励:
    _end_treasure_dists = _obs["feature"]
    _end_dist = _end_treasure_dists[0]
    reward += score
    reward += 0.7 * total_score
    reward = reward - 8*_end_dist
    #reward += 1*math.log(step+3)*(6-_end_dist)
    '''treasure_status = next_game_info["treasure_status"]
    for i in range(10):
        #if i==0:
        #    reward=reward+50
        if treasure_status[i] > 2:
            continue
        if treasure_status[i]==0:
            reward += 100
        elif treasure_status[i]==1:
            reward = reward - 8 * _end_treasure_dists[1+i]'''
    # 抵达终点的奖励
    #if terminated:
    #    reward += 350
        #reward += 3*score
    # 耗时惩罚
    #reward = reward - 5 * math.log(step+3)
    reward += - 0.5 * step
    #探索奖励
    #temp_location_memory=_extra_info["game_info"]["location_memory"]
    #print("treasure_status:",_extra_info["game_info"]["treasure_status"])
    #count = np.count_nonzero(temp_location_memory)
    #reward=reward+count*0.1
    #print("reward",reward)
    #print(reward)
    return reward
