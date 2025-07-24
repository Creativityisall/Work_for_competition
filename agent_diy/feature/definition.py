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
import torch

# 1. 修改SampleData，与参考代码对齐，使其只包含npdata
SampleData = create_cls("SampleData", npdata=None)

ObsData = create_cls("ObsData", feature=None, legal_actions=None, done=None)
ActData = create_cls("ActData", act=None)

class SampleManager:
    """
    新的数据收集器，用于替代Model内部的buffer。
    它在外部（训练流中）被创建和管理。
    """
    def __init__(self, gamma=0.99, gae_lambda=0.95):
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self._clear()

    def _clear(self):
        """清空所有列表"""
        self.state_seqs = []
        self.action_seqs = []
        self.actions = []
        self.logprobs = []
        self.values = []
        self.rewards = []
        self.dones = []
        self.advantages = [] # GAE优势
        self.returns = []    # GAE回报
        self.count = 0
        self.samples = [] # 用于存储处理后的SampleData

    def add(self, state_seq, action_seq, action, logprob, value, reward, done):
        """在每一步添加数据"""
        self.state_seqs.append(state_seq)
        self.action_seqs.append(action_seq)
        self.actions.append(action)
        self.logprobs.append(logprob)
        self.values.append(value)
        self.rewards.append(reward)
        self.dones.append(done)
        self.count += 1
    
    def process_last_frame(self, last_value, last_done):
        """
        在回合结束时调用。计算优势函数(GAE)和回报(Returns)。
        """
        values_np = torch.stack(self.values).squeeze().cpu().numpy()
        rewards_np = np.array(self.rewards, dtype=np.float32)
        dones_np = np.array(self.dones, dtype=np.float32)
        buffer_size = self.count

        advantages = np.zeros(buffer_size, dtype=np.float32)
        last_gae_lam = 0

        for step in reversed(range(buffer_size)):
            if step == buffer_size - 1:
                next_non_terminal = 1.0 - last_done
                next_values = last_value
            else:
                next_non_terminal = 1.0 - dones_np[step + 1]
                next_values = values_np[step + 1]
            
            delta = rewards_np[step] + self.gamma * next_values * next_non_terminal - values_np[step]
            last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
            advantages[step] = last_gae_lam

        returns = advantages + values_np
        
        self.advantages = advantages
        self.returns = returns
        
        # 调用_get_game_data将所有数据打包成npdata
        self.samples = self._get_game_data()


    # 2. 修改get_data，使其返回self.samples
    def get_data(self):
        """获取打包好的SampleData列表"""
        ret = self.samples
        self._clear() # 获取后清空
        return ret

    # 3. 新增_get_game_data方法，将所有数据打包成npdata
    def _get_game_data(self):
        """将收集到的所有数据转换为一个大的NumPy数组"""
        # 将Tensor列表转换为Numpy数组
        state_seqs_np = torch.stack(self.state_seqs).numpy()
        action_seqs_np = torch.stack(self.action_seqs).numpy()
        actions_np = torch.stack(self.actions).numpy()
        logprobs_np = torch.stack(self.logprobs).numpy()
        values_np = torch.stack(self.values).numpy()
        
        # 已经是Numpy的直接使用
        rewards_np = np.array(self.rewards, dtype=np.float32)
        dones_np = np.array(self.dones, dtype=np.float32)
        advantages_np = np.array(self.advantages, dtype=np.float32)
        returns_np = np.array(self.returns, dtype=np.float32)

        # 为了拼接，需要将所有数组调整为2D
        if len(state_seqs_np.shape) == 3: # (batch, seq, dim)
            state_seqs_flat = state_seqs_np.reshape(self.count, -1)
            action_seqs_flat = action_seqs_np.reshape(self.count, -1)
        
        # 确保其他1D数组也是2D的 (batch, 1)
        actions_2d = actions_np.reshape(self.count, -1)
        logprobs_2d = logprobs_np.reshape(self.count, -1)
        values_2d = values_np.reshape(self.count, -1)
        rewards_2d = rewards_np.reshape(self.count, -1)
        dones_2d = dones_np.reshape(self.count, -1)
        advantages_2d = advantages_np.reshape(self.count, -1)
        returns_2d = returns_np.reshape(self.count, -1)

        # 按照顺序拼接成一个大的np.array
        data = np.concatenate([
            state_seqs_flat, action_seqs_flat, actions_2d, logprobs_2d,
            values_2d, rewards_2d, dones_2d, advantages_2d, returns_2d
        ], axis=1)

        samples = []
        for i in range(self.count):
            samples.append(SampleData(npdata=data[i].astype(np.float32)))
        
        return samples

# --- 奖励函数部分保持不变 ---
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

    def reset(self,buff_count):
        self.visited_coordinates={}
        self.explored_treasure_pos=[]
        self.last_pos = None
        self.buff_remain = buff_count

class RewardConfig:
    hit_wall_punish = 30.0
    forget_buff_punish = 50.0
    each_step_punish = 1.5
    end_punish = 0.5
    treasure_dist_punish = 0.5
    revisit_punish_lowerbound = 3
    revisit_punish = 1
    treasure_reward = 50.0
    get_buff_reward = 30.0
    dist_reward_coef = 30.0
    end_reward = 200

def compute_distance(x, y):
    return math.sqrt((x[0] - y[0])**2 + (x[1] - y[1])**2)

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
    grid_size = 11
    for i in range(grid_size):
        for j in range(grid_size):
            if local_view_grid[i][j] == 4:
                absolute_pos = positions[i, j]
                treasure_absolute_positions.append(absolute_pos)
    return treasure_absolute_positions

def reward_shaping(rewardStateTracker, frame_no, terminated, truncated, obs, _obs, extra_info, _extra_info, step):
    map_info = _obs["map_info"]
    grid_size = 11
    local_view_grid = [map_info[i]["values"] for i in range(grid_size)]
    reward = 0
    reward = reward - RewardConfig.each_step_punish * step
    end_pos = (_extra_info["game_info"]["end_pos"]["x"], _extra_info["game_info"]["end_pos"]["z"])
    cur_pos = (_extra_info["game_info"]["pos"]["x"], _extra_info["game_info"]["pos"]["z"])
    end_dist = compute_distance(end_pos, cur_pos)
    reward = reward - RewardConfig.end_punish * end_dist
    if terminated:
        reward += RewardConfig.end_reward
        if rewardStateTracker.buff_remain > 0:
            reward = reward - RewardConfig.forget_buff_punish
    positions = get_position(local_view_grid, cur_pos)
    treasure_pos = get_treasure_position(positions, local_view_grid)
    rewardStateTracker.update_state(cur_pos, treasure_pos)
    if _extra_info["game_info"]["treasure_score"] > extra_info["game_info"]["treasure_score"]:
        for position in rewardStateTracker.explored_treasure_pos:
            if position!=(-1, -1) and position not in treasure_pos:
                rewardStateTracker.explored_treasure_pos.remove(position)
                rewardStateTracker.explored_treasure_pos.append((-1,-1))
    for position in rewardStateTracker.explored_treasure_pos:
        if position == (-1, -1):
            reward = reward + RewardConfig.treasure_reward
        else:
            reward = reward - RewardConfig.treasure_dist_punish * compute_distance(position, cur_pos)
    score = _obs["score_info"]["score"]
    reward = reward + score
    if _extra_info["game_info"]["buff_remain_time"] > 0:
        reward += RewardConfig.get_buff_reward
    for i in range(positions.shape[0]):
        for j in range(positions.shape[1]):
            view_grid_abs_pos = positions[i, j]
            visits_count = rewardStateTracker.visited_coordinates.get(view_grid_abs_pos, 0)
            if visits_count > RewardConfig.revisit_punish_lowerbound:
                reward -= RewardConfig.revisit_punish * visits_count
    if rewardStateTracker.last_pos == cur_pos:
        reward = reward - RewardConfig.hit_wall_punish
    rewardStateTracker.update_pos(cur_pos)
    return reward

# 4. 新增SampleData与NumpyData之间的转换函数
@attached
def SampleData2NumpyData(g_data):
    """从SampleData对象中提取npdata"""
    return g_data.npdata

@attached
def NumpyData2SampleData(s_data):
    """将npdata数组包装成SampleData对象"""
    return SampleData(npdata=s_data)