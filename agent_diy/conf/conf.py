#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2025 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


# Configuration of dimensions
# 关于维度的配置
class Config:
    GAMMA = 0.99
    GAE_LAMBDA = 0.9
    EPSILON = 0.15
    LR_PPO = 5e-4
    T_MAX = 75
    LOSS_WEIGHT = {'policy': 1.0, 'value': 0.5, 'entropy': 0.015}

    FEATURE_DIM = 88
    ACTION_DIM = 4
    LSTM_HIDDEN_SIZE = 256
    N_LSTM_LAYERS = 1
    LATENT_DIM_PI = 64
    LATENT_DIM_VF = 64

    BUFFER_SIZE = 10240
    N_ENVS = 1
    K_EPOCHS = 8
    UPDATE = 6144
    MINIBATCH = 2048

    # dimensionality of the sample
    # 样本维度
    SAMPLE_DIM = 6

    # Dimension of observation
    # 观察维度
    OBSERVATION_SHAPE = 250