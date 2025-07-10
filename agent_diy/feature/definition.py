#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2025 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


from kaiwu_agent.utils.common_func import create_cls, attached


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

def reward_shaping(frame_no, score, terminated, truncated, obs, _obs, step):
    reward = 0
    # 靠近终点的奖励:
    _end_treasure_dists = _obs["feature"]
    _end_dist = _end_treasure_dists[0]
    reward += - _end_dist
    # 抵达终点的奖励
    if terminated:
        reward += score
    # 耗时惩罚
    reward += - 0.01 * step
    return reward
