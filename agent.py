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

ObsData = create_cls("ObsData", feature=None, legal_actions=None)
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

    def reset(self):
        """重置智能体内部模型的状态"""
        if hasattr(self.model, 'reset'):
            self.model.reset()

    @predict_wrapper
    def predict(self, list_obs_data):
        state = list_obs_data[0].feature
        legal_actions = list_obs_data[0].legal_actions
        act = self.model.predict(state, legal_actions)
        return [ActData(act=act)]

    @exploit_wrapper
    def exploit(self, list_obs_data):
        obs_data=self.observation_process(list_obs_data["obs"], list_obs_data["extra_info"])
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
        return self.model.learn(list_sample_data)

    def observation_process(self, raw_obs, extra_info):
        game_info = extra_info["game_info"]
        pos = [game_info["pos_x"], game_info["pos_z"]]
        game_info = extra_info["game_info"]
        #print(f"DEBUG: raw_obs: {raw_obs}")
        #print(f"DEBUG: game_info: {game_info}")
        # 智能体当前位置相对于宝箱的距离(离散化)
        end_treasure_dists = raw_obs["feature"]

        # Feature #5: Graph features generation (obstacle information, treasure information, endpoint information)
        # 图特征生成(障碍物信息, 宝箱信息, 终点信息)
        #print("local_view:", game_info["local_view"])
        # 将一维视野转换为 5x5 的二维网格
        # 代理通常位于 5x5 网格的中心 (2, 2)
        grid_size = 5
        local_view_grid = [game_info["local_view"][i * grid_size : (i + 1) * grid_size] for i in range(grid_size)]

        # 代理在 5x5 网格中的中心坐标
        agent_row, agent_col = 2, 2

        legal_actions = []

        # 检查向上 (动作 0)
        # 向上移动意味着行索引减 1
        if agent_row - 1 >= 0:  # 确保不会超出网格上边界
            target_cell_value = local_view_grid[agent_row - 1][agent_col]
            # 如果目标单元格不是障碍物 (0)，则向上是合法动作
            if target_cell_value != 0:
                legal_actions.append(0)

        # 检查向下 (动作 1)
        # 向下移动意味着行索引加 1
        if agent_row + 1 < grid_size:  # 确保不会超出网格下边界
            target_cell_value = local_view_grid[agent_row + 1][agent_col]
            if target_cell_value != 0:
                legal_actions.append(1)

        # 检查向左 (动作 2)
        # 向左移动意味着列索引减 1
        if agent_col - 1 >= 0:  # 确保不会超出网格左边界
            target_cell_value = local_view_grid[agent_row][agent_col - 1]
            if target_cell_value != 0:
                legal_actions.append(2)

        # 检查向右 (动作 3)
        # 向右移动意味着列索引加 1
        if agent_col + 1 < grid_size:  # 确保不会超出网格右边界
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
        # 图特征转换为向量特征
        obstacle_flat, treasure_flat, end_flat = [], [], []
        for i in obstacle_map:
            obstacle_flat.extend(i)
        for i in treasure_map:
            treasure_flat.extend(i)
        for i in end_map:
            end_flat.extend(i)

        feature = np.concatenate(
            [
                pos,
                end_treasure_dists,
                obstacle_flat,
                treasure_flat,
                end_flat,
            ]
        )
        #print(feature)

        return ObsData(feature=feature,legal_actions= legal_actions)


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
