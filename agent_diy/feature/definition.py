#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2025 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""

import numpy as np
from kaiwu_agent.utils.common_func import create_cls, attached


SampleData = create_cls("SampleData", rewards=None, dones=None)
ObsData = create_cls("ObsData", feature=None, legal_actions=None)
ActData = create_cls("ActData", act=None)

@attached
def sample_process(list_game_data):
    rewards = []
    dones = []
    for sample in list_game_data:
        rewards.append(sample.reward)
        dones.append(sample.done)
    return SampleData(rewards=rewards, dones=dones)

pos_memory = []

def reward_shaping(frame_no, terminated, truncated, obs, next_obs, extra_info, next_extra_info):
    game_info, next_game_info = extra_info['game_info'], next_extra_info['game_info']
    next_pos_x, next_pos_z = next_game_info["pos_x"], next_game_info["pos_z"]
    end_treasure_dists, next_end_treasure_dists = obs["feature"], next_obs["feature"]
    next_local_view = next_game_info["local_view"]

    reward = 0
    # 发现宝箱的奖励
    if 4 in next_local_view:
        reward += 1
    # 发现终点的奖励
    if 3 in next_local_view:
        reward += 1
    # 靠近宝箱的奖励
    treasure_dist, next_treasure_dist = end_treasure_dists[1:],next_end_treasure_dists[1:]
    nearest_treasure_index = np.argmin(treasure_dist)
    if treasure_dist[nearest_treasure_index] > next_treasure_dist[nearest_treasure_index]:
        reward += 1
    else:
        # 远离宝箱的惩罚
        reward -= 1

    # 靠近终点的奖励
    end_dist, next_end_dist = end_treasure_dists[0], next_end_treasure_dists[0]
    if end_dist > next_end_dist:
        reward += 1

    # 重复探索或停止探索的惩罚
    if (next_pos_x, next_pos_z) in pos_memory:
        reward -= 1

    pos_memory.append((next_pos_x, next_pos_z))
    # 获得宝箱的奖励
    score = game_info['score']
    if score > 0 and not terminated:
        reward += score
    # 抵达终点的奖励
    if terminated:
        reward += score

    return reward
