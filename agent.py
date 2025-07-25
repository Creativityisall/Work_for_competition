#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2025 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


import torch
from kaiwu_agent.agent.base_agent import (
    predict_wrapper,
    exploit_wrapper,
    learn_wrapper,
    save_model_wrapper,
    load_model_wrapper,
    BaseAgent,
)

import random
import numpy as np
from kaiwu_agent.utils.common_func import attached
from agent_ppo.model.model import NetworkModelActor
from agent_ppo.algorithm.algorithm import Algorithm
from agent_ppo.feature.definition import SampleData, ObsData, ActData, SampleManager
from agent_ppo.feature.preprocessor import Preprocessor


def random_choice(log_p):
    p = np.exp(log_p - np.max(log_p))  # softmax
    p /= np.sum(p)
    r = random.random() * sum(p)
    s = 0
    for i in range(len(p)):
        if r > s and r <= s + p[i]:
            return i, p[i].item().log()
        s += p[i]
    return len(p) - 1, p[len(p) - 1].item().log()


@attached
class Agent(BaseAgent):
    def __init__(self, agent_type="player", device=None, logger=None, monitor=None):
        super().__init__(agent_type, device, logger, monitor)

        self.model = NetworkModelActor()
        self.algorithm = Algorithm(device=device, logger=logger, monitor=monitor)
        self.preprocessor = Preprocessor()
        self.sample_manager = SampleManager()
        self.win_history = []
        self.logger = logger
        self.reset()

    def update_win_rate(self, is_win):
        self.win_history.append(is_win)
        if len(self.win_history) > 100:
            self.win_history.pop(0)
        return sum(self.win_history) / len(self.win_history) if len(self.win_history) > 10 else 0

    def _predict(self, obs, legal_action):
        with torch.no_grad():
            inputs = self.model.format_data(obs, legal_action) # send an array to model's format_data method, which turns array into tensor directly
            output_list = self.model(*inputs) # forward pass, which returns lots of useless info for "Actor" (actor and learner share the same network structure)

        np_output_list = []
        for output in output_list:
            np_output_list.append(output.numpy().flatten())

        return np_output_list

    def predict_process(self, obs, legal_action):
        obs = np.array([obs]) # obs : dict -> array (only 1 element in array `obs`, which is a dictionary)
        legal_action = np.array([legal_action])
        log_probs, value = self._predict(obs, legal_action)
        return log_probs, value

    def observation_process(self, obs, extra_info=None):
        """
        基于当前帧观测信息+转移到当前帧的动作（上一动作），计算当前：
        1. 特征向量
        2. 合法动作
        3. 奖励
        并包装到 ObsData 中返回。
        
        注意：extra_info 不要用，因为 exploit 不能用它。只能提取 obs 里的信息。数据结构参考协议。
        """
        feature, legal_action, reward = self.preprocessor.process([obs, extra_info], self.last_action)

        return ObsData(
            feature=feature,
            legal_action=legal_action,
            reward=reward,
        )

    @predict_wrapper
    def predict(self, list_obs_data):
        feature = list_obs_data[0].feature
        legal_action = list_obs_data[0].legal_action
        log_probs, value = self.predict_process(feature, legal_action)

        assert log_probs.shape() == (1, len(legal_action)), "log_probs shape mismatch with legal_action"
        assert value.shape() == (1, 1), "value shape mismatch, should be (1, 1)"
        
        action, log_prob = random_choice(log_probs.squeeze(0).numpy())
        return [ActData(log_probs=log_probs, value=value, action=action, log_prob=log_prob)]

    def action_process(self, act_data):
        return act_data.action

    @exploit_wrapper
    def exploit(self, observation):
        obs_data = self.observation_process(observation["obs"], observation["extra_info"])
        feature = obs_data.feature
        legal_action = obs_data.legal_action
        log_probs, value = self.predict_process(feature, legal_action)
        action, log_prob = random_choice(log_probs)
        act = self.action_process(ActData(log_probs=log_probs, value=value, action=action, log_prob=log_prob))
        return act

    def reset(self):
        self.preprocessor.reset()
        self.last_prob = 0
        self.last_action = -1

    @learn_wrapper
    def learn(self, list_sample_data):
        self.algorithm.learn(list_sample_data)

    @save_model_wrapper
    def save_model(self, path=None, id="1"):
        # To save the model, it can consist of multiple files,
        # and it is important to ensure that each filename includes the "model.ckpt-id" field.
        # 保存模型, 可以是多个文件, 需要确保每个文件名里包括了model.ckpt-id字段
        model_file_path = f"{path}/model.ckpt-{str(id)}.pkl"
        torch.save(self.algorithm.model.state_dict(), model_file_path)
        self.logger.info(f"save model {model_file_path} successfully")

    @load_model_wrapper
    def load_model(self, path=None, id="1"):
        # When loading the model, you can load multiple files,
        # and it is important to ensure that each filename matches the one used during the save_model process.
        # 加载模型, 可以加载多个文件, 注意每个文件名需要和save_model时保持一致
        model_file_path = f"{path}/model.ckpt-{str(id)}.pkl"
        self.model.load_state_dict(torch.load(model_file_path, map_location="cpu"))
        self.logger.info(f"load model {model_file_path} successfully")
