#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2025 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""
import os
import numpy as np
import kaiwu_agent
from kaiwu_agent.agent.base_agent import BaseAgent
from kaiwu_agent.utils.common_func import create_cls, attached
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

ObsData = create_cls("ObsData", feature=None, legal_actions=None)
ActData = create_cls("ActData", action=None)

@attached
class Agent(BaseAgent):
    def __init__(self, agent_type="player", device=None, logger=None, monitor=None) -> None:
        super().__init__(agent_type, device, logger, monitor)
        self.model = Model(
            feature_dim = Config.FEATURE_DIM,
            action_dim = Config.ACTION_DIM,
            lstm_hidden_size = Config.LSTM_HIDDEN_SIZE,
            n_lstm_layers = Config.N_LSTM_LAYERS,
            latent_dim_pi = Config.LATENT_DIM_PI,
            latent_dim_vf = Config.LATENT_DIM_VF,
            gamma = Config.GAMMA,
            gae_lambda = Config.GAE_LAMBDA,
            eps_clip = Config.EPSILON,
            lr_ppo = Config.LR_PPO,
            step_size = Config.SCHEDULER_STEP_SIZE,
            lr_scheduler = Config.LR_SCHEDULER,
            loss_weight=Config.LOSS_WEIGHT,
            device = device,
            buffer_size = Config.BUFFER_SIZE,
            n_envs = Config.N_ENVS,
            K_epochs = Config.K_EPOCHS,
            minibatch = Config.MINIBATCH,
            logger=logger
        )
        self.logger = logger
        self.device = device
        self.monitor = monitor

    def reset(self):
        self.model.reset()

    def _features_extract(self, list_obs_data: list[ObsData]) -> tuple[np.ndarray, np.ndarray]:
        features = []
        legal_actions = []
        for obs_data in list_obs_data:
            features.append(obs_data.feature)
            legal_actions.append(obs_data.legal_actions)

        return np.array(features), np.array(legal_actions)

    @predict_wrapper
    def predict(self, list_obs_data: list[ObsData]) -> list[ActData]:
        features, legal_actions = self._features_extract(list_obs_data)

        list_act_data = []
        actions, log_probs = self.model.predict(features) # (n_envs, action_dim)
        self.logger.info(np.exp(log_probs.to('cpu')))
        for action in actions:
            act_data = ActData(action=action.to('cpu'))
            list_act_data.append(act_data)
        return list_act_data

    @exploit_wrapper
    def exploit(self, list_obs_data):
        obs, extra_info = list_obs_data["obs"], list_obs_data["extra_info"]
        list_obs_data=self.observation_process(list_obs=[obs], list_extra_info=[extra_info])
        features, legal_actions = self._features_extract(list_obs_data)

        actions = self.model.exploit(features) # (n_envs, action_dim)
        list_act_data = []
        for action in actions:
            act_data = ActData(action=action.to('cpu'))
            list_act_data.append(act_data)

        actions = self.action_process(list_act_data=list_act_data)
        return actions[0].item() #TODO: 分布式

    @learn_wrapper
    def learn(self, placeholder):
        self.model.learn()
        self.model.buffer.reset()
        print(self.model.buffer.pos)

    def action_process(self, list_act_data: list[ActData]) -> list[int]:
        actions = []
        for act_data in list_act_data:
            actions.append(act_data.action.item())
        return actions

    def collect(self, sample_data, list_obs_data) -> None:
        """采集环境反馈"""
        features, legal_actions = self._features_extract(list_obs_data)
        self.model.collect_rollouts(sample_data, features)

    def set_feature(self, list_obs_data):
        features, legal_actions = self._features_extract(list_obs_data)
        self.model.last_features = features

    def handle_timeout(self, truncateds, rewards, list_obs_data):
        features = []
        for obs_data in list_obs_data:
            features.append(obs_data.feature)

        return self.model.handle_timeout(truncateds, rewards, features)

    def compute_returns_and_advantage(self):
        """Compute value for the last timestep"""
        self.model.compute_returns_and_advantage()

    def collect_full(self) -> bool:
        """判断采样数据是否足够"""
        return self.model.buffer.full

    @save_model_wrapper
    def save_model(self, path=None, id="1"):
        self.model.save_model(path, id)

    @load_model_wrapper
    def load_model(self, path=None, id="1"):
        self.model.load_model(path, id)

    def _single_observation_process(self, obs, extra_info):
        game_info = extra_info["game_info"]
        # 合法动作
        local_view = game_info['local_view']
        legal_actions = []
        for i, view in enumerate([local_view[8], local_view[18], local_view[12], local_view[14]]): # UP DOWN LEFT RIGHT
            if view != 0:
                legal_actions.append(i)
        
        pos = [game_info["pos_x"], game_info["pos_z"]]

        # 智能体当前位置相对于宝箱的距离(离散化)
        end_treasure_dists = obs["feature"]

        # Feature #5: Graph features generation (obstacle information, treasure information, endpoint information)
        # 图特征生成(障碍物信息, 宝箱信息, 终点信息)
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
        
        obs_data = ObsData(feature=feature, legal_actions=legal_actions)
        return obs_data

    def observation_process(self, list_obs, list_extra_info):
        list_obs_data = []
        for obs, extra_info in zip(list_obs, list_extra_info):
            obs_data = self._single_observation_process(obs, extra_info)
            list_obs_data.append(obs_data)

        return list_obs_data