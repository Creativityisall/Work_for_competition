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
from kaiwu_agent.agent.preprocessor import single_observation_process
from agent_diy.conf.conf import Config
from agent_diy.model.model import Model

LSTMState = create_cls("LSTMState", pi=None, vf=None)
SampleData = create_cls("SampleData", rewards=None, dones=None)
ObsData = create_cls("ObsData", feature=None, legal_actions=None)
ActData = create_cls("ActData", action=None, prob=None)

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
            eps_clip = Config.EPSILON,
            lr_ppo = Config.LR_PPO,
            T_max=Config.T_MAX,
            loss_weight=Config.LOSS_WEIGHT,
            device = device,
            buffer_size = Config.BUFFER_SIZE,
            n_envs = Config.N_ENVS,
            K_epochs = Config.K_EPOCHS,
            minibatch = Config.MINIBATCH,
            mode = Config.TRAIN_MODE,
            logger=logger
        )
        # Advantage and Returns     
        self.gamma = Config.GAMMA # 奖励折扣因子，控制未来奖励的重要性
        self.gae_lambda = Config.GAE_LAMBDA # GAE（广义优势估计）的λ参数，平衡偏差和方差
        # Utilities
        self.logger = logger
        self.device = device
        self.monitor = monitor
        self.update_size = Config.UPDATE_SIZE

    def reset(self):
        self.model.reset()

    def handle_timeout(self, truncateds, rewards, list_obs_data):
        # TODO: 处理超时逻辑
        return rewards
    
    def compute_values(self, features: np.ndarray, lstm_hidden_state: LSTMState) -> np.ndarray:
        """Compute value for the last timestep"""
        return self.model.compute_values(features, lstm_hidden_state)

    def _features_extract(self, list_obs_data: list[ObsData]) -> tuple[np.ndarray, np.ndarray]:
        features = []
        legal_actions = []
        for obs_data in list_obs_data:
            features.append(obs_data.feature)
            legal_actions.append(obs_data.legal_actions)

        return np.array(features), np.array(legal_actions)
    
    def action_process(self, list_act_data: list[ActData]) -> list[int]:
        actions = []
        for act_data in list_act_data:
            actions.append(act_data.action.item())
        return actions

    def get_other_sample_data(self):
        """基于模型实现, 补充其它需要采样的数据"""
        return self.model.get_other_sample_data()

    def get_other_monitor_data(self):
        """基于监控需求, 补充其它需要监控的数据"""
        return self.model.get_other_monitor_data()

    @predict_wrapper
    def predict(self, list_obs_data: list[ObsData]) -> list[ActData]:
        features, legal_actions = self._features_extract(list_obs_data)

        list_act_data = []
        actions, log_probs = self.model.predict(features) # (n_envs, action_dim)
        for action, log_prob in zip(actions, log_probs):
            act_data = ActData(action=action, prob=np.exp(log_prob))
            # self.logger.info(f"{act_data.action} - {act_data.prob}")
            list_act_data.append(act_data)
        return list_act_data
    
    @exploit_wrapper
    def exploit(self, list_obs_data):
        obs, extra_info = list_obs_data["obs"], list_obs_data["extra_info"]
        list_obs_data=self.observation_process(list_obs=[obs], list_extra_info=[extra_info])
        features, legal_actions = self._features_extract(list_obs_data)

        actions, log_probs = self.model.exploit(features) # (n_envs, action_dim)
        list_act_data = []
        for action, log_prob in zip(actions, log_probs):
            act_data = ActData(action=action.to('cpu'), prob=np.exp(log_prob.to('cpu')))
            self.logger.info(f"{act_data.action} - {act_data.prob}")
            list_act_data.append(act_data)

        actions = self.action_process(list_act_data=list_act_data)
        return actions[0] #TODO: 分布式

    def observation_process(self, list_obs, list_extra_info):
        list_obs_data = []
        for obs, extra_info in zip(list_obs, list_extra_info):
            obs_data = single_observation_process(obs, extra_info)
            list_obs_data.append(obs_data)

        return list_obs_data
    
    @learn_wrapper
    def learn(self, list_sample_data: list[SampleData]) -> None:
        self.model.learn(list_sample_data)

    @save_model_wrapper
    def save_model(self, path=None, id="1"):
        self.model.save_model(path, id)

    @load_model_wrapper
    def load_model(self, path=None, id="1"):
        self.model.load_model(path, id)