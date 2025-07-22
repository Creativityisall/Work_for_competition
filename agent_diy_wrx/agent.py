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

ObsData = create_cls(
    "ObsData",
    feature=None,       # Used to design this way: (seq_len=1, feature_dim, ). Now, simply (feature_dim, ) 
    lstm_state_pi=None, # [0]: h, [1]: c, what shape ??
    lstm_state_vf=None, # [0]: h, [1]: c, shape ??
)

ActData = create_cls(           # s -> s' by action a with reward r and d'
    "ActData",
    action=None,                # a
    logprob=None,               # pi(a|s)
    entropy=None,               # 这步选择做出来时，四个动作概率分布的熵，用于后续计算 entropy loss
    value=None,                 # s 时的 value，训练时要算新的（value loss 要新旧对比）。workflow 里就是把 act_data 里记录的 value 填到这步转移的数据帧里的。
)



import torch 
from agent_diy.algorithm.algorithm import LstmPpoAlgorithm

from agent_diy.feature.preprocessor import RawObservation2FeatureVector

################ ABANDON THIS ↓ ############
class Agent(BaseAgent):
    def __init__(self, agent_type="player", device=None, logger=None, monitor=None) -> None:
        super().__init__(agent_type, device, logger, monitor)

    @predict_wrapper
    def predict(self, list_obs_data):
        """
        [obs_data], [other_info] -> [ActData]
        """

        pass

    @exploit_wrapper
    def exploit(self, observation):
        """
        observation -> [acts]

        空白框架给的输入格式是，但是 q-learning 示例代码和手册都说是单个 observation （dict类型）环境观测。
        输出是动作列表，应该只要包含一个动作即可（列表是为了兼容王者3v3，单环境多动作）。
        """



        pass

    @learn_wrapper
    def learn(self, list_sample_data):
        """
        [SampleData] -> None

        Aisvr收集数据时调用 learn，输入是收集好的一轮 rollout 的 Frames 组成的列表。输送到全局 Mempool 中。

        Learner 学习时循环调用 learn，输入是 Mempool 中的 SampleData 列表。列表大小即 batch_size，在 configure_app.toml 中定义。
        这里只要实现学习逻辑即可。
        """


        pass

    def observation_process(self, obs, extra_info):
        """
        Extract features from observation data.
        
        `obs` and `extra_info` are current state raw information.
        Combine with `history` to process observation into lstm_state.

        Ouput should be an ObsData object with a lstm_state attribute of length `Config.N_SEQ`.
        """        
        return ObsData()

    # def action_process(self, act_data):
        # return act_data.act

    @save_model_wrapper
    def save_model(self, path=None, id="1"):
        pass

    @load_model_wrapper
    def load_model(self, path=None, id="1"):
        pass
################ ABANDON THIS ↑ ############

class Agent(BaseAgent):
    def __init__(self, agent_type="player", device=None, logger=None, monitor=None) -> None:
        self.device = device
        self.logger = logger
        self.monitor = monitor

        self.algo = LstmPpoAlgorithm(
            config=Config,
            device=self.device,
            logger=self.logger,
            monitor=self.monitor,
        )

    def observation_process(self, obs, extra_info) -> ObsData: # type: ignore
        # NOTE 更多的特征处理，可以在 preprossessor.py 里开发，然后在这里汇总成 ObsData 对象返回提交。
        feature = RawObservation2FeatureVector(obs, extra_info)
        current_lstm_state_pi, current_lstm_state_vf = self.algo.get_current_lstm_hidden_state()

        # 严格遵守 API：要返回 ObsData 类型对象（按照手册，要用 create_cls 函数定义）
        return ObsData(
            feature=feature, 
            lstm_state_pi=current_lstm_state_pi, 
            lstm_state_vf=current_lstm_state_vf
        )
    
    # TODO 读完手册后补充实现这个本来看似冗余的方法。游戏变复杂，要重新设计动作空间，或许真的需要解包 ActData 函数。
    # def action_process(self, act_data: ActData) -> int:
    #     return ActData.action
        

    @predict_wrapper    
    def predict(self, list_obs_data) -> [ActData]: # type: ignore
        """
        输入是一个 ObsData 类型元素的列表（之前原始观测已经被 observation_process 处理过；其实只有一个元素），输出是一个 ActData 类型元素的列表（其实只有一个）。
        
        该方法调用 agent.forward_a_step(..)，其中参数只有 ObsData 列表，得到动作相关信息的各自列表。
        为由于该函数在 rollout 阶段被调用，需要随机采样动作，故 actions = None, deterministic=False
        """
        actions, logprobs, entropies, values = self.algo.forward_a_step(
            list_obs_data=list_obs_data, 
            list_action=None,
            deterministic=False
        )
        
        return [
            ActData(
                action=action,
                logprob=logprob,
                entropy=entropy,
                value=value
            )
            for action, logprob, entropy, value in zip(actions, logprobs, entropies, values)
        ]



    @exploit_wrapper
    def exploit(self, observation) -> list[int]:
        """
        输入是一个原始数据（包括obs和extra_info），输出是一个动作列表（其实只含有一个动作）

        该方法调用 agent.forward_a_step(..)，其中参数为 ObsData 列表 + 确定性选择动作。返回值为动作相关信息的各自列表。
        为由于该函数在评估阶段被调用，需要贪心采样动作，故 actions = None, deterministic=True
        """
        obs, extra_info = observation["obs"], observation["extra_info"]
        ObsData = self.observation_process(obs, extra_info)
        actions, _, _, _ = self.algo.forward_a_step(list_obs_data=[ObsData], deterministic=True)
        action = actions[0].item()  # actions 是一个一维 tensor (batch_size=1, )，取第一个元素并转换为 int
        return action 

    @load_model_wrapper
    def load_model(self, path=None, id="latest"):
        # To load the model, it can consist of multiple files,
        # and it is important to ensure that each filename includes the "model.ckpt-id" field.
        # 加载模型, 可以是多个文件, 需要确保每个文件名里包括了model.ckpt-id字段
        model_file_path = f"{path}/model.ckpt-{str(id)}.pkl"

        # Load the model's state dictionary from the CPU
        # 从CPU加载模型的状态字典
        model_state_dict_cpu = torch.load(model_file_path, map_location=self.device)
        self.algo.load_model(model_state_dict_cpu)

    @save_model_wrapper
    def save_model(self, path=None, id="latest"):
        # To save the model, it can consist of multiple files,
        # and it is important to ensure that each filename includes the "model.ckpt-id" field.
        # 保存模型, 可以是多个文件, 需要确保每个文件名里包括了model.ckpt-id字段
        model_file_path = f"{path}/model.ckpt-{str(id)}.pkl"

        # Copy the model's state dictionary to the CPU
        # 将模型的状态字典拷贝到CPU
        model_state_dict_cpu = self.algo.snapshot_model()
        torch.save(model_state_dict_cpu, model_file_path)

    
    @learn_wrapper
    def learn(self, list_sample_data):
        # XXX 我猜 Learner 可能会实例化一个 Agent 类对象，并且自动调用 agent.load_model() 方法加载模型池里最新模型。但此猜测有一个缺陷：如果是这个机制，则无法加载指定模型，这可能是有人不希望的。所以，我还是手动加载一下模型。
        # self.load_model(id="latest")  # Load latest model
        self.algo.learn(list_sample_data)
        # self.save_model()