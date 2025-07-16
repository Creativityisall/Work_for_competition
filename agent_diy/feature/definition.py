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
ActData = create_cls("ActData", action=None)


@attached
def sample_process(sample):
    return SampleData(rewards=sample.rewards, dones=sample.dones)

class Map:
    def __init__(self, map_size=64):
        self.map = np.full((map_size, map_size), -1) 
        self.end = None
        self.treasures = ()

    def record_local_view(self, pos, view, view_half=2):
        x, y = pos
        view_rec = np.asarray(view).reshape(5, 5)
        # 计算视野窗口在整张地图上的行/列范围 以及 处理边界
        H, W = self.map.shape
        r0, r1 = max(x - view_half, 0), min(x + view_half + 1, H)   # 行范围
        c0, c1 = max(y - view_half, 0), min(y + view_half + 1, W)   # 列范围
        # 计算未探索的空间
        local_map = self.map[r0:r1, c0:c1]
        unknown = np.sum(local_map == -1)
        # 记录终点与宝箱的位置
        treasure_mask = (view == 4)
        if treasure_mask:
            treasure_rows, treasure_cols = np.where(treasure_mask)
            abs_treasure_x = treasure_rows + r0   # 行
            abs_treasure_y = treasure_cols + c0   # 列
            self.treasures.update(set(zip(abs_treasure_x, abs_treasure_y)))

        end_mask = (view == 3)
        if end_mask:
            end_row, end_col = np.where(end_mask)
            abs_end_x = end_rows + r0   # 行
            abs_end_y = end_cols + c0   # 列
            self.end = list(zip(abs_end_x, abs_end_y))

        # 复制地图
        self.map[r0:r1, c0:c1] = view_rec
        return unknown


map = Map()

def single_reward_shaping(frame_no, terminated, truncated, obs, next_obs, extra_info, next_extra_info, step):
    game_info, next_game_info = extra_info['game_info'], next_extra_info['game_info']
    next_pos_x, next_pos_z = next_game_info["pos_x"], next_game_info["pos_z"]
    end_treasure_dists, next_end_treasure_dists = obs["feature"], next_obs["feature"]
    next_local_view = next_game_info["local_view"]
    unkonwn = map.record_local_view((next_pos_x, next_pos_z), next_local_view)

    reward = 0

    # # 探索奖励
    # if unkonwn > 0:
    #     reward += 10
    # else: # 重复惩罚
    #     reward -= 10

    # # 靠近宝箱的奖励
    # treasure_dist, next_treasure_dist = end_treasure_dists[1:],next_end_treasure_dists[1:]
    # nearest_treasure_index = np.argmin(treasure_dist)
    # if treasure_dist[nearest_treasure_index] > next_treasure_dist[nearest_treasure_index]:
    #     reward += 10
    # else:
    #     # 远离宝箱的惩罚
    #     reward -= 10

    # # 靠近终点的奖励
    # end_dist, next_end_dist = end_treasure_dists[0], next_end_treasure_dists[0]
    # if end_dist > next_end_dist:
    #     reward += 10

    # 获得宝箱的奖励
    score = game_info['score']
    if score > 0 and not terminated:
        reward += score
    # 抵达终点的奖励
    if terminated:
        reward += score

    return reward

def reward_shaping(
    list_frame_no, 
    list_terminated, 
    list_truncated, 
    list_obs, 
    list_next_obs, 
    list_extra_info, 
    list_next_extra_info, 
    step
    ) -> list[int]:
    rewards = []
    for idx in range(len(list_frame_no)):
        reward = single_reward_shaping(
            list_frame_no[idx], 
            list_terminated[idx], 
            list_truncated[idx], 
            list_obs[idx], 
            list_next_obs[idx], 
            list_extra_info[idx], list_next_extra_info[idx], 
            step
        )
        rewards.append(reward)

    return rewards