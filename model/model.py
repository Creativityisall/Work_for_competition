#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2025 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


from typing import List
import torch
from torch import nn
import numpy as np
from agent_ppo.conf.conf import Config

import sys
import os

if os.path.basename(sys.argv[0]) == "learner.py":
    import torch

    torch.set_num_interop_threads(2)
    torch.set_num_threads(2)
else:
    import torch

    torch.set_num_interop_threads(4)
    torch.set_num_threads(4)


####################################
# from agent_ppo.agent import my_trace
####################################



class NetworkModelBase(nn.Module):
    def __init__(self):
        super().__init__()
        # feature configure parameter
        # 特征配置参数
        self.data_split_shape = Config.DATA_SPLIT_SHAPE
        self.feature_split_shape = Config.FEATURE_SPLIT_SHAPE
        self.label_size = Config.ACTION_NUM
        self.feature_len = Config.FEATURE_LEN
        self.value_num = Config.VALUE_NUM

        self.var_beta = Config.BETA_START
        self.vf_coef = Config.VF_COEF

        self.clip_param = Config.CLIP_PARAM

        self.data_len = Config.data_len

        self.cnn_input_dim = 128 * 128
        self.cnn_output_dim = 256

        # Main MLP network
        # 主MLP网络
        self.cnn_net = nn.Sequential(
            # 将128*128展开为128x128的单通道图像
            nn.Unflatten(-1, (1, 128, 128)),
            
            # 第一个卷积层
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 64x64
            
            # 第二个卷积层
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 32x32
            
            # 第三个卷积层
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 16x16
            
            # 展平并全连接
            nn.Flatten(),
            nn.Linear(128 * 16 * 16, 512),
            nn.ReLU(),
            nn.Linear(512, self.cnn_output_dim),
            nn.ReLU()
        )
        self.main_fc_dim_list = [self.feature_len - self.cnn_input_dim + self.cnn_output_dim, 128, 256]

        self.main_mlp_net = MLP(self.main_fc_dim_list, "main_mlp_net", non_linearity_last=True)
        self.label_mlp = MLP([256, 64, self.label_size], "label_mlp")
        self.value_mlp = MLP([256, 64, self.value_num], "value_mlp")

    def process_legal_action(self, label, legal_action):
        label_max, _ = torch.max(label * legal_action, 1, True)
        label = label - label_max
        label = label * legal_action
        label = label + 1e5 * (legal_action - 1) # illegal action will be set to a very large negative value
        return label

    def forward(self, feature, legal_action):
        # Slice out the CNN input
        # 切片出CNN输入
        front_feature = feature[:, :self.cnn_input_dim]
        cnn_feature = feature[:, -self.cnn_input_dim:]

        # CNN processing
        # CNN处理
        cnn_feature = self.cnn_net(cnn_feature)

        # Concatenate CNN output with the rest of the feature
        combined_feature = torch.cat([front_feature, cnn_feature], dim=1)
        
        # Main MLP processing
        # 主MLP处理
        fc_mlp_out = self.main_mlp_net(combined_feature)

        # Action and value processing
        # 处理动作和值
        label_mlp_out = self.label_mlp(fc_mlp_out)
        label_out = self.process_legal_action(label_mlp_out, legal_action)

        log_prob = torch.nn.functional.log_softmax(label_out, dim=1)
        value = self.value_mlp(fc_mlp_out)

        return log_prob, value # 这里自动支持批处理


class NetworkModelActor(NetworkModelBase):
    def format_data(self, obs, legal_action):
        return (
            torch.tensor(obs).to(torch.float32),
            torch.tensor(legal_action).to(torch.float32),
        )


class NetworkModelLearner(NetworkModelBase):
    def format_data(self, datas):
        return datas.view(-1, self.data_len).float().split(self.data_split_shape, dim=1)

    def forward(self, data_list, inference=False):
        feature = data_list[0]
        legal_action = data_list[-1]        
        return super().forward(feature, legal_action)


def make_fc_layer(in_features: int, out_features: int):
    # Wrapper function to create and initialize a linear layer
    # 创建并初始化一个线性层
    fc_layer = nn.Linear(in_features, out_features)

    # initialize weight and bias
    # 初始化权重及偏移量
    nn.init.orthogonal(fc_layer.weight)
    nn.init.zeros_(fc_layer.bias)

    return fc_layer


class MLP(nn.Module):
    def __init__(
        self,
        fc_feat_dim_list: List[int],
        name: str,
        non_linearity: nn.Module = nn.ReLU,
        non_linearity_last: bool = False,
    ):
        # Create a MLP object
        # 创建一个 MLP 对象
        super().__init__()
        self.fc_layers = nn.Sequential()
        for i in range(len(fc_feat_dim_list) - 1):
            fc_layer = make_fc_layer(fc_feat_dim_list[i], fc_feat_dim_list[i + 1])
            self.fc_layers.add_module("{0}_fc{1}".format(name, i + 1), fc_layer)
            # no relu for the last fc layer of the mlp unless required
            # 除非有需要，否则 mlp 的最后一个 fc 层不使用 relu
            if i + 1 < len(fc_feat_dim_list) - 1 or non_linearity_last:
                self.fc_layers.add_module("{0}_non_linear{1}".format(name, i + 1), non_linearity())

    def forward(self, data):
        return self.fc_layers(data)
