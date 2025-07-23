#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2025 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""

from kaiwu_agent.utils.common_func import create_cls, attached
#from agent_diy.feature.preprocessor import Preprocessor
import numpy as np
import math

SampleData = create_cls("SampleData", state_seqs=None, action_seqs=None, actions=None, logprobs=None, values=None, dones=None, rewards=None, done=None, last_state=None, episode=None)
ObsData = create_cls("ObsData", feature=None, legal_actions=None, done=None)
ActData = create_cls("ActData", act=None)

class RewardStateTracker:
    def __init__(self, buff_count):
        self.visited_coordinates = {} # 使用set来存储唯一坐标，效率高，防止重复
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

@attached
def sample_process(model, list_game_data, gamma, last_state, episode):
    rewards = []
    dones = []
    discounted_reward = 0
    for sample in list_game_data:
        rewards.append(sample.reward)
        dones.append(sample.done)
        # discounted_reward = sample.reward + (gamma * discounted_reward)
        # rewards.insert(0, discounted_reward)
    return SampleData(state_seqs=model.buffer['state_seqs'], action_seqs=model.buffer['action_seqs'], actions=model.buffer['actions'], logprobs=model.buffer['logprobs'],
    values=model.buffer['values'], dones=model.buffer['dones'], rewards=rewards, done=dones, last_state=last_state, episode=episode)

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

class RewardConfig:
    # 惩罚项系数
    hit_wall_punish = 30.0   # 普通行走撞墙的惩罚
    forget_buff_punish = 50.0     # 结束时未拾取Buff的惩罚
    each_step_punish = 1.5        # 每一步的固定惩罚（时间惩罚），在训练后期引入
    end_punish = 0.5
    treasure_dist_punish = 0.5
    revisit_punish_lowerbound = 3
    revisit_punish = 1
    # 奖励项系数
    treasure_reward = 50.0
    get_buff_reward = 30.0        # 获得Buff的奖励
    dist_reward_coef = 30.0       # 普通移动时，靠近目标的距离奖励系数
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

def reward_shaping(rewardStateTracker, frame_no, score, terminated, truncated, obs, _obs, extra_info, _extra_info, step):
    map_info = _obs["map_info"]
    grid_size = 11
    local_view_grid = [map_info[i]["values"] for i in range(grid_size)]
    reward = 0
    #1.步数惩罚
    reward = reward - RewardConfig.each_step_punish * step

    #2.到终点的距离惩罚
    end_pos = (_extra_info["game_info"]["end_pos"]["x"], _extra_info["game_info"]["end_pos"]["z"])
    cur_pos = (_extra_info["game_info"]["pos"]["x"], _extra_info["game_info"]["pos"]["z"])
    end_dist = compute_distance(end_pos, cur_pos)
    reward = reward - RewardConfig.end_punish * end_dist
    #奖励
    if terminated:
        reward += RewardConfig.end_reward
        #如果忘记拾取buff，给予惩罚
        if rewardStateTracker.buff_remain > 0:#TODO:怎么判断剩余buff的数量
            reward = reward - RewardConfig.forget_buff_punish
    positions = get_position(local_view_grid, cur_pos)
    treasure_pos = get_treasure_position(positions, local_view_grid)

    rewardStateTracker.update_state(cur_pos, treasure_pos)

    if _extra_info["game_info"]["treasure_score"] > extra_info["game_info"]["treasure_score"]:
        for position in rewardStateTracker.explored_treasure_pos:
            if position!=(-1, -1) and position not in treasure_pos:#TODO
                rewardStateTracker.explored_treasure_pos.remove(position)
                rewardStateTracker.explored_treasure_pos.append((-1,-1))

    #3.到各个宝箱的距离惩罚
    for position in rewardStateTracker.explored_treasure_pos:
        if position == (-1, -1):
            reward = reward + RewardConfig.treasure_reward
        else:
            reward = reward - RewardConfig.treasure_dist_punish * compute_distance(position, cur_pos)

    #4,5.拾取宝箱、到达终点score
    reward = reward + score

    #6.拾取buffer奖励
    if _extra_info["game_info"]["buff_remain_time"] > 0:
        reward += RewardConfig.get_buff_reward

    #7.重复位置惩罚
    for i in range(positions.shape[0]):
        for j in range(positions.shape[1]):
            view_grid_abs_pos = positions[i, j] 
            
            visits_count = rewardStateTracker.visited_coordinates.get(view_grid_abs_pos, 0)
            
            if visits_count > RewardConfig.revisit_punish_lowerbound:
                reward -= RewardConfig.revisit_punish * visits_count

    #8.撞墙惩罚
    if rewardStateTracker.last_pos == cur_pos:
        reward = reward - RewardConfig.hit_wall_punish

    rewardStateTracker.update_pos(cur_pos)

    return reward

@attached
def sample_process(model, list_game_data, gamma, last_state, episode):
    """
    数据处理函数。在打包数据前，检查缓冲区是否为空。
    """
    #print(model.buffer)
    # 如果模型缓冲区为空，则没有可训练的数据，返回None
    if not model.buffer or not model.buffer['actions']:
        return None
    
    rewards = []
    dones = []
    for sample in list_game_data:
        rewards.append(sample.reward)
        dones.append(sample.done)

    # 从model.buffer中提取所有需要的数据，打包成SampleData对象
    # 注意：这里的dones是从list_game_data来的，而model.buffer['dones']是torch.tensor，我们需要保持一致
    # model.learn方法中使用的是 model.buffer['dones']，所以我们传递那个
    return SampleData(
        state_seqs=model.buffer['state_seqs'],
        action_seqs=model.buffer['action_seqs'],
        actions=model.buffer['actions'],
        logprobs=model.buffer['logprobs'],
        values=model.buffer['values'],
        dones=model.buffer['dones'],  # 使用buffer中的dones
        rewards=rewards,
        last_state=last_state,
        last_done=dones[-1], # last_done应对应最后一帧的done状态
        episode=episode
    )


@attached
def SampleData2NumpyData(g_data):
    """
    将包含完整训练数据的SampleData对象序列化为一个numpy数组。
    """
    # 如果g_data为None (由sample_process返回)，则直接返回None
    if g_data is None:
        return None
    
    # 获取动态的buffer_size, 如果actions为空也返回None
    if not g_data.actions:
        return None
    buffer_size = len(g_data.actions)
    
    # 元数据: buffer_size, episode, last_done
    # 将last_done转为浮点数
    last_done_float = getattr(g_data, 'last_done', False)
    meta_data = np.array([buffer_size, g_data.episode, float(last_done_float)], dtype=np.float32)

    # 转换所有数据为numpy数组并扁平化
    state_seqs_np = torch.stack(g_data.state_seqs).numpy().flatten()
    action_seqs_np = torch.stack(g_data.action_seqs).numpy().flatten()
    actions_np = torch.stack(g_data.actions).numpy().flatten()
    logprobs_np = torch.stack(g_data.logprobs).numpy().flatten()
    values_np = torch.stack(g_data.values).numpy().flatten()
    # dones也来自buffer，是tensor列表
    dones_np = torch.stack(g_data.dones).numpy().flatten() 
    rewards_np = np.array(g_data.rewards, dtype=np.float32).flatten()
    last_state_np = np.array(g_data.last_state, dtype=np.float32).flatten()
    
    return np.hstack(
        (
            meta_data, state_seqs_np, action_seqs_np, actions_np,
            logprobs_np, values_np, dones_np, rewards_np, last_state_np
        )
    )
