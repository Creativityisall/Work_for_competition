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

SampleData = create_cls("SampleData", rewards=None, done= None, last_state=None, episode = None)
ObsData = create_cls("ObsData", feature=None, legal_actions=None, done = None)
ActData = create_cls("ActData", act=None)

@attached
def sample_process(list_game_data, gamma, last_state, episode):
    rewards = []
    dones = []
    discounted_reward = 0
    for sample in list_game_data:
        rewards.append(sample.reward)
        dones.append(sample.done)
        #discounted_reward = sample.reward + (gamma * discounted_reward)
        #rewards.insert(0, discounted_reward)
        
    return SampleData(rewards=rewards, done = dones, last_state= last_state, episode = episode)

def find_nearest_active_treasure_info(dists, status):
    """
    一个辅助函数，用于找到最近的、未被拾取的宝箱的索引和距离。
    - dists: 到所有宝箱的距离列表 (不含终点)
    - status: 宝箱的状态列表 (例如，0=未拾取, 1=已拾取)
    """
    min_dist = float('inf')
    nearest_idx = -1
    
    # 宝箱距离从 obs["feature"][1:] 开始
    treasure_dists = dists[1:]

    for i in range(len(treasure_dists)):
        # 假设状态0代表宝箱存在且可拾取
        if status[i] == 0 and treasure_dists[i] < min_dist:
            min_dist = treasure_dists[i]
            nearest_idx = i
            
    # 返回的是在宝箱列表中的索引(0-9)和最小距离
    return nearest_idx, min_dist


def reward_shaping(frame_no, score, terminated, truncated, obs, _obs, extra_info, _extra_info, step):
    
    # --- 1. 基础奖励和惩罚 ---
    # 事件奖励（捡到宝箱、到达终点时的瞬时分数）
    reward = score
    # 时间惩罚，鼓励效率
    #reward -= 0.5 * step  # 每走一步都付出固定代价

    # --- 2. 基于进度的塑形奖励 (核心部分) ---
    prev_dists = obs["feature"]
    curr_dists = _obs["feature"]
    
    prev_treasure_status = extra_info['game_info']["treasure_status"]
    curr_treasure_status = _extra_info['game_info']["treasure_status"]

    # 检查是否还有未拾取的宝箱
    # `all(s > 0 for s in curr_treasure_status)` 是一种可能的判断方式
    # 或者根据你的环境，可能有更直接的字段，比如 treasure_count
    treasures_left = (len(curr_treasure_status) - np.count_nonzero(curr_treasure_status)) > 0
    
    # 设定奖励系数
    C_PROGRESS = 25.0  # 靠近目标的奖励系数

    if treasures_left > 7:
        # **阶段一：还有宝箱没捡，目标是最近的宝箱**
        prev_nearest_idx, prev_nearest_dist = find_nearest_active_treasure_info(prev_dists, prev_treasure_status)
        curr_nearest_idx, curr_nearest_dist = find_nearest_active_treasure_info(curr_dists, curr_treasure_status)

        # 确保我们找到了一个有效的宝箱作为目标
        if prev_nearest_idx != -1 and curr_nearest_idx != -1:
            # 如果智能体捡到了一个宝箱，我们比较的是到新目标的距离，这可能导致prev_dist < curr_dist
            # 所以，只在目标宝箱没变的情况下给予靠近奖励
            if prev_nearest_idx == curr_nearest_idx:
                progress = prev_nearest_dist - curr_nearest_dist
                reward += C_PROGRESS * progress
            # (可选) 如果捡到了一个宝箱，可以给一个额外的发现新目标的奖励
            elif score > 0:
                 reward += 20 # 捡到宝箱后，鼓励它快速定位下一个目标
        
        treasure_status = curr_treasure_status
        for i in range(10):
            reward = reward - 6 * _end_treasure_dists[1+i]
        temp_location_memory=_extra_info["game_info"]["location_memory"]
        count = np.count_nonzero(temp_location_memory)
        reward=reward + count * 1
    
    else:
        reward = reward + 200
        # **阶段二：宝箱捡完，目标是终点**
        prev_end_dist = prev_dists[0]
        curr_end_dist = curr_dists[0]
        
        progress = prev_end_dist - curr_end_dist
        reward += C_PROGRESS * progress
        reward = reward - 20 * curr_end_dist
        
    return reward
'''def reward_shaping(frame_no, score, terminated, truncated, obs, _obs, extra_info, _extra_info, step):
    reward = 0
    #print(_obs["feature"])
    game_info, next_game_info = extra_info['game_info'], _extra_info['game_info']
    next_pos_x, next_pos_z = next_game_info["pos_x"], next_game_info["pos_z"]
    end_treasure_dists, next_end_treasure_dists = obs["feature"], _obs["feature"]
    next_local_view = next_game_info["local_view"]
    ''
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
        reward += 30''
    total_score = next_game_info["total_score"]
    treasure_score = next_game_info["treasure_score"]
    treasure_count = next_game_info["treasure_count"]
    # 靠近终点的奖励:
    _end_treasure_dists = _obs["feature"]
    _end_dist = _end_treasure_dists[0]
    #reward += score
    #reward += 0.9 * total_score
    reward = reward - 10*_end_dist
    #reward += 1*math.log(step+3)*(6-_end_dist)
    treasure_status = next_game_info["treasure_status"]
    for i in range(10):
        #if i==0:
        #    reward=reward+50
        if treasure_status[i] > 2:
            continue
        if treasure_status[i]==0:
            reward += 100
        elif treasure_status[i]==1:
            reward = reward - 6 * _end_treasure_dists[1+i]
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
    #reward=reward+count*0.5
    #print("reward",reward)
    #print(reward)
    return reward'''
