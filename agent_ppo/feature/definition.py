#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2025 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""

from agent_ppo.conf.conf import Config
from kaiwu_agent.utils.common_func import create_cls, attached
from agent_ppo.model.model import DualLSTM
import numpy as np
import math

# The create_cls function is used to dynamically create a class.
# The first parameter of the function is the type name, and the remaining parameters are the attributes of the class.
# The default value of the attribute should be set to None.
# create_cls函数用于动态创建一个类，函数第一个参数为类型名称，剩余参数为类的属性，属性默认值应设为None
ObsData = create_cls(
    "ObsData",
    feature=None,
    state=None,
    legal_action=None,
    reward=None,
)

ActData = create_cls(
    "ActData",
    probs=None,
    value=None,
    target=None,
    predict=None,
    action=None,
    prob=None,
)

SampleData = create_cls("SampleData", npdata=None)

RelativeDistance = {
    "RELATIVE_DISTANCE_NONE": 0,
    "VerySmall": 1,
    "Small": 2,
    "Medium": 3,
    "Large": 4,
    "VeryLarge": 5,
}

RelativeDirection = {
    "East": 1,
    "NorthEast": 2,
    "North": 3,
    "NorthWest": 4,
    "West": 5,
    "SouthWest": 6,
    "South": 7,
    "SouthEast": 8,
}

DirectionAngles = {
    1: 0,
    2: 45,
    3: 90,
    4: 135,
    5: 180,
    6: 225,
    7: 270,
    8: 315,
}


class RewardStateTracker:
    def __init__(self, buff_count):
        self.visited_coordinates = {}
        self.explored_treasure_pos = []
        self.last_pos = None
        self.buff_remain = buff_count

    def update_state(self, current_pos, treasure_pos):
        self.visited_coordinates[current_pos] = self.visited_coordinates.get(current_pos, 0) + 1
        for position in treasure_pos:
            if position not in self.explored_treasure_pos:
                self.explored_treasure_pos.append(position)

    def update_pos(self, pos):
        self.last_pos = pos

    def reset(self, buff_count):
        self.visited_coordinates = {}
        self.explored_treasure_pos = []
        self.last_pos = None
        self.buff_remain = buff_count


class RewardConfig:
    hit_wall_punish = 30.0
    forget_buff_punish = 50.0
    each_step_punish = 1.5
    end_punish = 5
    treasure_dist_punish = 0.5
    revisit_punish_lowerbound = 3
    revisit_punish = 1
    treasure_reward = 50.0
    get_buff_reward = 30.0
    dist_reward_coef = 30.0
    end_reward = 200
    explored_reward = 2
    grid_size = 11


def compute_distance(x, y):
    return math.sqrt((x[0] - y[0]) ** 2 + (x[1] - y[1]) ** 2)


def get_position(local_view_grid, cur_pos) -> np.ndarray:
    grid_size = len(local_view_grid[0])
    positions = np.empty((grid_size, grid_size), dtype=object)
    center_x, center_y = cur_pos
    top_left_x = center_x - 5
    top_left_y = center_y - 5
    for i in range(grid_size):
        for j in range(grid_size):
            abs_x = top_left_x + j
            abs_y = top_left_y + i
            positions[i, j] = (abs_x, abs_y)
    return positions


def get_treasure_position(positions, local_view_grid):
    treasure_absolute_positions = []
    grid_size = RewardConfig.grid_size
    for i in range(grid_size):
        for j in range(grid_size):
            if local_view_grid[i][j] == 4:
                absolute_pos = positions[i, j]
                treasure_absolute_positions.append(absolute_pos)
    return treasure_absolute_positions


def reward_process(rewardStateTracker, step, terminated, truncated, obs, _obs, extra_info, _extra_info, 
                   end_dist, history_dist, end_pos):  # TODO:考虑end_dist,history_dist以及done
    if not rewardStateTracker:
        return 0
    map_info = _obs["map_info"]
    grid_size = 11
    local_view_grid = [map_info[i]["values"] for i in range(grid_size)]
    reward = 0
    # 1.步数惩罚
    reward = reward - RewardConfig.each_step_punish * step

    # 2.到终点的距离惩罚
#    end_pos = (_extra_info["game_info"]["end_pos"]["x"], _extra_info["game_info"]["end_pos"]["z"])
    cur_pos = (_extra_info["game_info"]["pos"]["x"], _extra_info["game_info"]["pos"]["z"])
    end_dist = compute_distance(end_pos, cur_pos)
    reward = reward - RewardConfig.end_punish * end_dist
    # 奖励
    #    if terminated:
    #        reward += RewardConfig.end_reward
    #        #如果忘记拾取buff，给予惩罚
    #        if rewardStateTracker.buff_remain > 0:#TODO:怎么判断剩余buff的数量
    #            reward = reward - RewardConfig.forget_buff_punish
    positions = get_position(local_view_grid, cur_pos)
    treasure_pos = get_treasure_position(positions, local_view_grid)

    rewardStateTracker.update_state(cur_pos, treasure_pos)

    if _extra_info["game_info"]["treasure_score"] > extra_info["game_info"]["treasure_score"]:
        grid_size = RewardConfig.grid_size
        for i in range(grid_size):
            for j in range(grid_size):
                if local_view_grid[i][j] != 4 and (
                i, j) in rewardStateTracker.explored_treasure_pos:  # TODO:这里对于不用超级闪现的情况是适用的，否则还需调整
                    rewardStateTracker.explored_treasure_pos.remove(position)
                    rewardStateTracker.explored_treasure_pos.append((-1, -1))

    # 3.到各个宝箱的距离惩罚
    for position in rewardStateTracker.explored_treasure_pos:
        if position == (-1, -1):
            reward = reward + RewardConfig.treasure_reward
        else:
            reward = reward - RewardConfig.treasure_dist_punish * compute_distance(position, cur_pos)

    # 4,5.拾取宝箱、到达终点score
    reward = reward + _extra_info["game_info"]["score"]

    # 6.拾取buffer奖励
    if _extra_info["game_info"]["buff_remain_time"] > 0:
        reward += RewardConfig.get_buff_reward

    # 7.重复位置惩罚
    for i in range(positions.shape[0]):
        for j in range(positions.shape[1]):
            view_grid_abs_pos = positions[i, j]

            visits_count = rewardStateTracker.visited_coordinates.get(view_grid_abs_pos, 0)

            if visits_count > RewardConfig.revisit_punish_lowerbound:
                reward -= RewardConfig.revisit_punish * visits_count

    # 8.撞墙惩罚
    if rewardStateTracker.last_pos == cur_pos:
        reward = reward - RewardConfig.hit_wall_punish

    rewardStateTracker.update_pos(cur_pos)

    # 9.终点惩罚
    #end_reward = -RewardConfig.end_punish * end_dist
    #reward += end_reward

    # 10.距离奖励
    #explore_reward = RewardConfig.explored_reward * history_dist
    #reward += explore_reward

    return [reward]


class SampleManager:
    def __init__(
            self,
            gamma=0.99,
            tdlambda=0.95,
    ):
        # self.dual_lstm = DualLSTM(state_input_dim, output_dim)
        self.gamma = Config.GAMMA
        self.tdlambda = Config.TDLAMBDA

        self.feature = []
        self.probs = []
        self.actions = []
        self.reward = []
        self.value = []
        self.adv = []
        self.tdlamret = []
        self.legal_action = []
        self.count = 0
        self.samples = []
        self.state_seqs = []
        self.action_seqs = []

    def add(self, feature, legal_action, prob, action, value, reward):  # TODO:state_seqs,action要不要加？
        self.feature.append(feature)
        self.legal_action.append(legal_action)
        self.probs.append(prob)
        self.actions.append(action)
        self.value.append(value)
        self.reward.append(reward)
        self.adv.append(np.zeros_like(value))
        self.tdlamret.append(np.zeros_like(value))
        self.count += 1

    def add_last_reward(self, reward):
        self.reward.append(reward)
        self.value.append(np.zeros_like(reward))

    def update_sample_info(self):
        last_gae = 0
        for i in range(self.count - 1, -1, -1):
            reward = self.reward[i + 1]
            next_val = self.value[i + 1]
            val = self.value[i]
            delta = reward + next_val * self.gamma - val
            last_gae = delta + self.gamma * self.tdlambda * last_gae
            self.adv[i] = last_gae
            self.tdlamret[i] = last_gae + val

    def sample_process(self, feature, legal_action, prob, action, value, reward):
        self.add(feature, legal_action, prob, action, value, reward)

    def process_last_frame(self, reward):
        self.add_last_reward(reward)
        # 发送前的后向传递更新
        # Backward pass updates before sending
        self.update_sample_info()
        # self.state_seqs = self.dual_lstm(self.feature, self.action_seqs)
        self.samples = self._get_game_data()

    def get_game_data(self):
        ret = self.samples
        self.samples = []
        return ret

    def _get_game_data(self):
        feature = np.array(self.feature).transpose()
        probs = np.array(self.probs).transpose()
        actions = np.array(self.actions).transpose()
        reward = np.array(self.reward[:-1]).transpose()
        value = np.array(self.value[:-1]).transpose()
        legal_action = np.array(self.legal_action).transpose()
        adv = np.array(self.adv).transpose()
        tdlamret = np.array(self.tdlamret).transpose()
        # state_seqs = np.array(self.state_seqs).transpose()

        data = np.concatenate([feature, reward, value, tdlamret, adv, actions, probs, legal_action]).transpose()

        samples = []
        for i in range(0, self.count):
            samples.append(SampleData(npdata=data[i].astype(np.float32)))

        return samples


@attached
def SampleData2NumpyData(g_data):
    return g_data.npdata


@attached
def NumpyData2SampleData(s_data):
    return SampleData(npdata=s_data)
