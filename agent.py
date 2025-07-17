#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2025 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""

import kaiwu_agent
from kaiwu_agent.agent.base_agent import BaseAgent
from kaiwu_agent.utils.common_func import create_cls
import numpy as np
import os
from kaiwu_agent.agent.base_agent import (
    learn_wrapper,
    save_model_wrapper,
    load_model_wrapper,
    predict_wrapper,
    exploit_wrapper,
    check_hasattr,
)
from agent_diy.conf.conf import Config
from agent_diy.model.model import Model
import torch

ObsData = create_cls("ObsData", feature=None, legal_actions=None, done = None)
ActData = create_cls("ActData", act=None)


class Agent(BaseAgent):
    def __init__(self, agent_type="player", device=None, logger=None, monitor=None) -> None:
        super().__init__(agent_type, device, logger, monitor)
        self.model = Model(
            input_dim = Config.INPUT_SIZE,
            output_dim = Config.OUTPUT_SIZE,
            seq_length = Config.LSTM_SEQ_LENGTH,
            lr_actor = Config.LR_ACTOR,
            lr_critic = Config.LR_CRITIC,
            lr_lstm = Config.LR_LSTM,
            eps_clip = Config.EPSILON,
            K_epochs = Config.K_STEPS,
            loss_weight = Config.LOSS_WEIGHT,
            lstm_hidden_dim = Config.LSTM_HIDDEN_SIZE,
            lstm_num_layers = Config.LSTM_HIDDEN_LAYERS
        )
        self.gamma = Config.GAMMA
        self._POS_MEANS = np.array([32.0, 32.0], dtype=np.float32)
        self._POS_STDS = np.array([10.67, 10.67], dtype=np.float32)

        # 相对距离 (end_treasure_dists) 范围: [0, 6]
        # 假设 raw_obs["feature"] 只有一个元素 (即只有一个距离值)
        self._END_TREASURE_DISTS_MEANS = np.array([3.0], dtype=np.float32)
        self._END_TREASURE_DISTS_STDS = np.array([1.0], dtype=np.float32)
    def reset(self):
        """重置智能体内部模型的状态"""
        if hasattr(self.model, 'reset'):
            self.model.reset()

    @predict_wrapper
    def predict(self, list_obs_data):
        state = list_obs_data[0].feature
        legal_actions = list_obs_data[0].legal_actions
        act = self.model.predict(state, list_obs_data[0].done, legal_actions)
        return [ActData(act=act)]

    @exploit_wrapper
    def exploit(self, list_obs_data):
        done=False
        if list_obs_data["extra_info"]["game_info"]["local_view"][12] == 3:
            done=True
        obs_data=self.observation_process(done, list_obs_data["obs"], list_obs_data["extra_info"])
        state = obs_data.feature
        legal_actions=obs_data.legal_actions
        action = self.model.exploit(state,legal_actions)
        ##
        #action = torch.tensor([2])
        ##return [ActData(act=action.detach())]
        ##
        act = action.detach().cpu().item()
        return act

    @learn_wrapper
    def learn(self, list_sample_data):
        return self.model.learn(list_sample_data , list_sample_data.last_state, list_sample_data.done[-1])

    import numpy as np
    import torch  # 假设这里会用到torch，尽管目前代码片段没有直接使用
    from collections import namedtuple

    # 定义一个简单的ObsData结构，以便函数能够返回
    ObsData = namedtuple("ObsData", ["feature", "legal_actions", "done"])

    # --- Z-score 归一化统计量 (请根据您的实际数据进行替换!) ---
    # 假设 pos_x 和 pos_z 的范围可能是相对较大，比如 0 到 100
    # 假设 end_treasure_dists 可能是 0 到 100 左右的距离
    # 这些值应该从您的训练数据集中计算得出！
    # 例如：
    # mean_pos_x, std_pos_x = 50.0, 20.0
    # mean_pos_z, std_pos_z = 50.0, 20.0
    # mean_end_treasure_dists = np.array([50.0, 50.0, 50.0]) # 假设有3个距离特征
    # std_end_treasure_dists = np.array([20.0, 20.0, 20.0])

    # 示例统计量 - **请务必替换为您的实际数据统计量！**
    # 如果您不确定，可以先运行几轮环境收集一些数据，然后计算这些统计量。
    # 例如，如果 pos_x, pos_z 范围是 [0, 100]，end_treasure_dists 范围是 [0, 50]，
    # 那么简单的估计可以是：
    # mean = (min + max) / 2
    # std = (max - min) / 6 (基于经验法则，覆盖99.7%数据在均值±3标准差内)
    # 所以 pos 维度 (假设两个坐标都在 [0,100]): mean=50, std=33.3 (100/3)
    # end_treasure_dists 维度 (假设在 [0,50]): mean=25, std=16.6 (50/3)

    # 假设 pos 有 2 个维度，end_treasure_dists 有 3 个维度 (请根据实际情况调整)
    # 您需要根据 raw_obs["feature"] 的实际维度来确定 end_treasure_dists_means/stds 的长度
    # 如果 raw_obs["feature"] 只有一个元素，那么只需要一个 mean 和 std
    _POS_MEANS = np.array([50.0, 50.0], dtype=np.float32)
    _POS_STDS = np.array([33.3, 33.3], dtype=np.float32)

    # 假设 end_treasure_dists 只有 1 个维度，即 raw_obs["feature"] 的长度是 1
    # 如果 raw_obs["feature"] 实际上是多个距离的列表，请调整这里的维度和均值/标准差
    _END_TREASURE_DISTS_MEANS = np.array([25.0], dtype=np.float32)  # 假设它是一个长度为1的数组
    _END_TREASURE_DISTS_STDS = np.array([16.6], dtype=np.float32)

    # --- 归一化函数 ---
    def _normalize_feature(self, value, mean, std):
        """ 对特征进行 Z-score 归一化，作为内部辅助方法 """
        # 避免除以0，如果标准差为0，则保持原样或设为0 (意味着该维度是常数)
        # 将为0的标准差替换为1，避免除零并确保计算可行
        if np.any(std == 0):
            std = np.where(std == 0, 1.0, std)
            # 可以在这里添加日志或警告，提醒用户某些特征的标准差为零
            # print("Warning: Standard deviation is zero for some features during normalization. Check your data.")
        return (value - mean) / std

    def observation_process(self, done, raw_obs, extra_info):
        game_info = extra_info["game_info"]
        pos = np.array([game_info["pos_x"], game_info["pos_z"]], dtype=np.float32)

        # 智能体当前位置相对于宝箱的距离(离散化)
        # 确保 end_treasure_dists 也是 np.array 类型
        end_treasure_dists = np.array(raw_obs["feature"], dtype=np.float32)

        # --- 对 pos 和 end_treasure_dists 进行归一化，使用类成员变量 ---
        normalized_pos = self._normalize_feature(pos, self._POS_MEANS, self._POS_STDS)
        normalized_end_treasure_dists = self._normalize_feature(end_treasure_dists, self._END_TREASURE_DISTS_MEANS, self._END_TREASURE_DISTS_STDS)
        # --- 归一化结束 ---

        # Feature #5: Graph features generation (obstacle information, treasure information, endpoint information)
        # 图特征生成(障碍物信息, 宝箱信息, 终点信息)
        grid_size = 5
        local_view_grid = [game_info["local_view"][i * grid_size : (i + 1) * grid_size] for i in range(grid_size)]

        agent_row, agent_col = 2, 2

        legal_actions = []

        # 检查向上 (动作 0)
        if agent_row - 1 >= 0:
            target_cell_value = local_view_grid[agent_row - 1][agent_col]
            if target_cell_value != 0:
                legal_actions.append(0)

        # 检查向下 (动作 1)
        if agent_row + 1 < grid_size:
            target_cell_value = local_view_grid[agent_row + 1][agent_col]
            if target_cell_value != 0:
                legal_actions.append(1)

        # 检查向左 (动作 2)
        if agent_col - 1 >= 0:
            target_cell_value = local_view_grid[agent_row][agent_col - 1]
            if target_cell_value != 0:
                legal_actions.append(2)

        # 检查向右 (动作 3)
        if agent_col + 1 < grid_size:
            target_cell_value = local_view_grid[agent_row][agent_col + 1]
            if target_cell_value != 0:
                legal_actions.append(3)

        local_view = [game_info["local_view"][i : i + 5] for i in range(0, len(game_info["local_view"]), 5)]
        obstacle_map, treasure_map, end_map = [], [], []
        for sub_list in local_view:
            obstacle_map.append([1 if i == 0 else 0 for i in sub_list])
            treasure_map.append([1 if i == 4 else 0 for i in sub_list])
            end_map.append([1 if i == 3 else 0 for i in sub_list])

        # Feature #6: Conversion of graph features into vector features
        obstacle_flat, treasure_flat, end_flat = [], [], []
        for i in obstacle_map:
            obstacle_flat.extend(i)
        for i in treasure_map:
            treasure_flat.extend(i)
        for i in end_map:
            end_flat.extend(i)

        feature = np.concatenate(
            [
                normalized_pos,  # 使用归一化后的位置
                normalized_end_treasure_dists,  # 使用归一化后的宝箱距离
                obstacle_flat,
                treasure_flat,
                end_flat,
            ]
        )

        return ObsData(feature=feature, legal_actions=legal_actions, done=done)

    def action_process(self, act_data):
        return act_data.act.item()

    @save_model_wrapper
    def save_model(self, path=None, id="1"):
        self.model.save_model(path, id)

    @load_model_wrapper
    def load_model(self, path=None, id="1"):
        model_filename = f"model.ckpt-{id}.pt"
        full_model_path = os.path.join(path, model_filename)

        self.model.load_model(full_model_path)
        self.reset()
