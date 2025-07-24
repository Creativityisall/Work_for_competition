#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2025 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from kaiwu_agent.utils.common_func import attached


class Model(nn.Module):
    def __init__(self, state_shape, action_shape):
        super().__init__()

        # User-defined network
        # 用户自定义网络



def _layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class FeatureEncoderModel(nn.Module):
    def __init__(
            self, 
            device, 
            logger, monitor,
            input_dim,
            output_dim,
        ):
        super().__init__()
        self.device = device
        self.logger = logger
        self.monitor = monitor

        # Feature Encoder Network Parameters
        self.input_dim = input_dim
        self.output_dim = output_dim

        # FeatureEncoder network structure.
        # NOTE For now, actor and critic share the same feature encoder network.
        self.featureEncoder_network = nn.Sequential(
            _layer_init(nn.Linear(self.input_dim, 128)),
            nn.Tanh(), # or nn.ReLU(), which is better?
            nn.Linear(128, self.output_dim)
        )


    def forward(self, x):
        """
        Forward pass through the feature encoder network.

        x: (b, feature_dim, )  # 输入特征向量，形状为 (batch_size, feature_dim)
        返回值： (b, feature_encoded_dim, )  # 输出 encode 后的特征向量
        """
        return self.featureEncoder_network(x)
    
    def snapshot_model(self):
        """获取当前模型的快照"""
        # 返回当前 feature_encoder_network 的状态字典（网络参数）
        return self.featureEncoder_network.state_dict()
    
    def load_model(self, model_state_dict_cpu):
        """
        Load the model state dictionary from CPU.
        """
        # 解读出 feature_encoder_network 网络的参数并加载。
        self.featureEncoder_network.load_state_dict(model_state_dict_cpu)



class LstmModel(nn.Module):
    def __init__(
            self, 
            config, 
            device, 
            logger, 
            monitor, 

            pi_input_size,
            pi_hidden_size,
            pi_num_layers,
            vf_input_size,
            vf_hidden_size,
            vf_num_layers,
            
            seq_len,
        ):
        super().__init__()
        self.config = config
        self.device = device
        self.logger = logger
        self.monitor = monitor

        self.pi_input_size = pi_input_size
        self.pi_hidden_size = pi_hidden_size
        self.pi_num_layers = pi_num_layers
        self.vf_input_size = vf_input_size
        self.vf_hidden_size = vf_hidden_size
        self.vf_num_layers = vf_num_layers

        self.seq_len = seq_len

        
# TODO 如何更好地初始化 LSTM 网络？
        """
        1. 根据传入参数，定义两个 LSTM 网络的结构

        rollout 阶段：
        - 正常の输入：   (seq_len=1,  batch_size=1, *_input_size)
        - 隐藏层输入：   (num_layers, batch_size=1, *_hidden_size)
        - 输出：        (seq_len=1,  batch_size=1, *_hidden_size)

        learn 阶段： batch_size = configure_app.toml 中的 buffer_size  
        """

        # 好在初始化网络和 batch_size 无关 :)
        self.lstm_pi_network = nn.LSTM(
            input_size=self.pi_input_size,
            hidden_size=self.pi_hidden_size,
            num_layers=self.pi_num_layers,
        )

        self.lstm_vf_network = nn.LSTM(
            input_size=self.vf_input_size,
            hidden_size=self.vf_hidden_size,
            num_layers=self.vf_num_layers,
        )

        """
        2. 初始化两个网络共4个隐藏层

        每种隐藏层的形状规定为 (num_layers, batch_size=1, *_hidden_size)。这样可以直接送入 LSTM 网络进行前向传播。

        NOTE 因为 LstmModel 对象记录的当前隐藏层，是服务于 rollout 阶段的，所以 batch_size=1。
        NOTE learn 阶段的 lstm 输入全部来自 ObsData，其过网所带来的隐藏层变化，已经不用记录了（off-policy）。此时批数据格式由 algo.forward_a_step() 保证对齐。
        """
        self.current_hidden_state_pi = [
            np.zeros((self.pi_num_layers, 1, self.pi_hidden_size), dtype=np.float32),
            np.zeros((self.pi_num_layers, 1, self.pi_hidden_size), dtype=np.float32)
        ]
        self.current_hidden_state_vf = [
            np.zeros((self.vf_num_layers, 1, self.vf_hidden_size), dtype=np.float32),
            np.zeros((self.vf_num_layers, 1, self.vf_hidden_size), dtype=np.float32)
        ]


    def pi_forward(self, x, hidden_state):
        """
        Forward pass through the LSTM networks.
        - x: 输入数据，形状为 (seq_len, batch_size, input_size)
        - hidden_state: 隐藏状态，形状为 (num_layers, batch_size, hidden_size)

        返回值：
        - output: LSTM 网络的输出，形状为 (seq_len, batch_size, hidden_size)
        - new_hidden_state: 更新后的隐藏状态，形状为 (num_layers, batch_size, hidden_size)
        """
        output, new_hidden_state = self.lstm_pi_network(x, hidden_state)
        return output, new_hidden_state

    def vf_forward(self, x, hidden_state):
        """
        Forward pass through the value function LSTM network.
        - x: 输入数据，形状为 (seq_len, batch_size, input_size)
        - hidden_state: 隐藏状态，形状为 (num_layers, batch_size, hidden_size)

        返回值：
        - output: LSTM 网络的输出，形状为 (seq_len, batch_size, hidden_size)
        - new_hidden_state: 更新后的隐藏状态，形状为 (num_layers, batch_size, hidden_size)
        """
        output, new_hidden_state = self.lstm_vf_network(x, hidden_state)
        return output, new_hidden_state

    def get_current_hidden_state(self):
        """
        返回值当前的隐藏态，形状：
        - current_hidden_state_pi: [h, c]，每个是 (num_layers, batch_size=1, pi_hidden_size)
        - current_hidden_state_vf: [h, c]，每个是 (num_layers, batch_size=1, vf_hidden_size)
        """
        return self.current_hidden_state_pi, self.current_hidden_state_vf


    def snapshot_model(self):
        """获取当前模型的快照"""
        # 返回当前两个 LSTM 网络的状态字典（网络参数）。
        # NOTE 当前隐藏层，不需要快照？毕竟之后不会再加载。
        return {
            "lstm_pi_network": self.lstm_pi_network.state_dict(),
            "lstm_vf_network": self.lstm_vf_network.state_dict(),
        }
    
    def load_model(self, model_state_dict_cpu):
        # 解读出 lstm_pi_network 和 lstm_vf_network 网络的参数并加载。
        self.lstm_pi_network.load_state_dict(model_state_dict_cpu["lstm_pi_network"])
        self.lstm_vf_network.load_state_dict(model_state_dict_cpu["lstm_vf_network"])
        
        # NOTE 该方法在每轮 rollout 开始之前被调用，所以加载完网络后，还要初始化 current_hidden_state_pi 和 current_hidden_state_vf 为 相应形状的 0。
        self.current_hidden_state_pi = [
            np.zeros((self.pi_num_layers, 1, self.pi_hidden_size), dtype=np.float32),
            np.zeros((self.pi_num_layers, 1, self.pi_hidden_size), dtype=np.float32)
        ]
        self.current_hidden_state_vf = [
            np.zeros((self.vf_num_layers, 1, self.vf_hidden_size), dtype=np.float32),
            np.zeros((self.vf_num_layers, 1, self.vf_hidden_size), dtype=np.float32)
        ]
        


class PpoModel(nn.Module):
    def __init__(
        self, 
        config, 
        device, 
        logger, 
        monitor,

        # network related parameters
        policy_net_input_dim,
        policy_net_output_dim,
        value_net_input_dim,    
    ):
        super().__init__()
        self.config = config
        self.device = device
        self.logger = logger
        self.monitor = monitor
        
        self.value_net_input_dim = value_net_input_dim
        self.policy_net_input_dim = policy_net_input_dim
        self.policy_net_output_dim = policy_net_output_dim
        
        # NOTE NO NEED for algo-related parameters here (Algorithm class will handle them).
         

        # Build up the policy and value networks.
        self.value_network = nn.Sequential(
            _layer_init(nn.Linear(self.value_net_input_dim, 64)),
            nn.Tanh(),
            _layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            _layer_init(nn.Linear(64, 1), std=1.0),
        )

        self.policy_network = nn.Sequential(
            _layer_init(nn.Linear(self.policy_net_input_dim, 64)),
            nn.Tanh(),
            _layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            _layer_init(nn.Linear(64, self.policy_net_output_dim), std=0.01),
        )

    def policy_forward(self, pi_input_batch, vf_input_batch, actions=None, deterministic=False):
        """
        Forward pass (mainly) through the policy network.
        NOTE also involves the value network.

        - pi_input_batch: 输入数据，形状为 (batch_size, policy_input_dim)
        - vf_input_batch: 输入数据，形状为 (batch_size, value_input_dim)
        - actions: 可选参数，指定动作，形状为 (batch_size, )。如果为 None，则根据概率分布采样/取最大概率动作。
            - 若 deterministic=True，则取最大概率动作。
            - 若 deterministic=False，则从概率分布中采样动作。

        返回值：
        - output: 策略网络的输出，形状为 (batch_size, policy_output_dim)
        """
        logits = self.policy_network(pi_input_batch)
        probs = torch.distributions.categorical.Categorical(logits=logits)
        if actions == None:
            if deterministic:
                actions = probs.mode()
            else:
                actions = probs.sample()

        return actions, probs.log_prob(actions), probs.entropy(), self.value_network(vf_input_batch)

    def snapshot_model(self):
        """获取当前模型的快照"""
        # 返回当前 Policy 和 Value 网络的状态字典（网络参数）。
        return {
            "value_network": self.value_network.state_dict(),
            "policy_network": self.policy_network.state_dict(),
        }
    
    def load_model(self, model_state_dict_cpu):
        # 解读出 lstm_pi_network 和 lstm_vf_network 网络的参数并加载。
        self.value_network.load_state_dict(model_state_dict_cpu["value_network"])
        self.policy_network.load_state_dict(model_state_dict_cpu["policy_network"])
