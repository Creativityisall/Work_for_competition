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
from agent_diy.feature.definition import sample_process
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
import numpy as np
from collections import namedtuple

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
        self._POS_MEANS = np.array([63.5, 63.5], dtype=np.float32)
        self._POS_STDS = np.array([36.66, 36.66], dtype=np.float32)
        self.sample_process = sample_process

        # 相对距离 (end_treasure_dists) 范围: [0, 6]
        # 假设 raw_obs["feature"] 只有一个元素 (即只有一个距离值)
        #self._END_TREASURE_DISTS_MEANS = np.array([3.0], dtype=np.float32)
        #self._END_TREASURE_DISTS_STDS = np.array([1.0], dtype=np.float32)
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
    def learn(self, list_game_data):
        """
        这个方法在Actor端被调用，用于在数据收集后打包数据。
        真正的 self.model.learn 在Learner端被调用。
        """
        # 从 list_game_data 中提取 learn 方法所需的参数
        # 注意: list_game_data 在这里是原始的、由 predict 逐步收集的样本列表
        last_state = list_game_data[-1].obs.feature
        episode = list_game_data[-1].episode

        # 调用我们修改后的 sample_process，并传入 self.model
        sample_data_obj = self.sample_process(
            self.model, 
            list_game_data, 
            self.gamma, 
            last_state, 
            episode
        )
        
        # sample_process 现在可能会返回 None，所以 learn wrapper 需要返回这个值
        # 框架的序列化/反序列化流程会处理 None
        return sample_data_obj

    # --- 归一化函数 ---
    def _normalize_feature(self, value, mean, std):
        if np.any(std == 0):
            std = np.where(std == 0, 1.0, std)
        return (value - mean) / std

    def _normalize_feature(self, feature, means, stds):
        # Simple normalization based on pre-calculated means and stds
        return (feature - means) / (stds + 1e-8)

    def observation_process(self, done, raw_obs, extra_info):
        game_info = extra_info["game_info"]
        map_info = raw_obs["map_info"]
        pos = np.array([game_info["pos"]["x"], game_info["pos"]["z"]], dtype=np.float32)

        # --- Normalizing pos and end_treasure_dists using class member variables ---
        normalized_pos = self._normalize_feature(pos, self._POS_MEANS, self._POS_STDS)
        # --- Normalization end ---

        # Feature #5: Graph features generation (obstacle information, treasure information, endpoint information)
        # Graph feature generation (obstacle information, treasure information, endpoint information)
        grid_size = 11
        # Reconstruct the 2D local view grid
        local_view_grid = [map_info[i]["values"] for i in range(grid_size)]

        agent_row, agent_col = 5, 5 # Assuming agent is always at the center of the 11x11 local view

        legal_actions = []

        move_directions = [(0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1)]

        for action_idx, (dr, dc) in enumerate(move_directions):
            target_row, target_col = agent_row + dr, agent_col + dc

            # Check if target cell is within grid boundaries
            if 0 <= target_row < grid_size and 0 <= target_col < grid_size:
                target_cell_value = local_view_grid[target_row][target_col]
                # Assuming value 0 indicates an obstacle (impassable)
                if target_cell_value != 0:
                    legal_actions.append(action_idx)

        legal_act = raw_obs["legal_act"]
        if legal_act[1] == 1:
            legal_actions.extend(list(range(8, 16)))

        # Feature #6: Conversion of graph features into vector features
        # local_view is already created above as local_view_grid
        obstacle_map, treasure_map, end_map, buff_map = [], [], [], []
        for r_idx in range(grid_size):
            obstacle_row = []
            treasure_row = []
            end_row = []
            buff_row = []
            for c_idx in range(grid_size):
                cell_value = local_view_grid[r_idx][c_idx]
                obstacle_row.append(1 if cell_value == 0 else 0)
                treasure_row.append(1 if cell_value == 4 else 0)
                end_row.append(1 if cell_value == 3 else 0)
                buff_row.append(1 if cell_value == 6 else 0)
            obstacle_map.append(obstacle_row)
            treasure_map.append(treasure_row)
            end_map.append(end_row)
            buff_map.append(buff_row)

        obstacle_flat, treasure_flat, end_flat, buff_flat= [], [], [], []
        for row in obstacle_map:
            obstacle_flat.extend(row)
        for row in treasure_map:
            treasure_flat.extend(row)
        for row in end_map:
            end_flat.extend(row)
        for row in buff_map:
            buff_flat.extend(row)

        feature = np.concatenate(
            [
                normalized_pos,  # Using normalized position
                obstacle_flat,
                treasure_flat,
                end_flat,
                buff_flat
            ]
        )
        #print("length", len(feature)) # Uncomment for debugging
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
