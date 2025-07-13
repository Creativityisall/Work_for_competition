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

SampleData = create_cls("SampleData", rewards=None)
ObsData = create_cls("ObsData", feature=None, legal_actions=None)
ActData = create_cls("ActData", act=None)

@attached
def sample_process(list_game_data, gamma):
    rewards = []
    discounted_reward = 0
    for sample in reversed(list_game_data):
        if sample.done:
            discounted_reward = 0
        discounted_reward = sample.reward + (gamma * discounted_reward)
        rewards.insert(0, discounted_reward)
        
    return SampleData(rewards=rewards)

def reward_shaping(frame_no, score, terminated, truncated, obs, _obs, _extra_info, step):
    reward = 0
    #print(_obs["feature"])
    # 靠近终点的奖励:
    _end_treasure_dists = _obs["feature"]
    _end_dist = _end_treasure_dists[0]
    reward += - 10*_end_dist
    #reward += 1*math.log(step+3)*(6-_end_dist)
    for i in _end_treasure_dists[1:]:
        if i==0:
            reward=reward+50
        reward=reward-6*i
    # 抵达终点的奖励
    if terminated:
        reward += 100
        #reward += 3*score
    # 耗时惩罚
    reward = reward - 0.1* math.log(step+3)
    #reward += - 0.5 * step
    #探索奖励
    temp_location_memory=_extra_info["game_info"]["location_memory"]
    #print("treasure_status:",_extra_info["game_info"]["treasure_status"])
    count = np.count_nonzero(temp_location_memory)
    reward=reward+count*0.1
    #print("reward",reward)
    #print(reward)
    return reward
