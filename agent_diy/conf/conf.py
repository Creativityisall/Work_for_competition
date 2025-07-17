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
    GAMMA = 0.95
    GAE_LAMBDA = 0.95
    EPSILON = 0.1
    LR_PPO = 1e-5
    SCHEDULER_STEP_SIZE = 10
    LR_SCHEDULER = 0.1
    LOSS_WEIGHT = {'policy': 1.0, 'value': 0.5, 'entropy': 0.015}

    FEATURE_DIM = 88
    ACTION_DIM = 4
    LSTM_HIDDEN_SIZE = 256
    N_LSTM_LAYERS = 1
    LATENT_DIM_PI = 64
    LATENT_DIM_VF = 64

    BUFFER_SIZE = 3072
    N_ENVS = 1
    K_EPOCHS = 10
    MINIBATCH = 1024

    # dimensionality of the sample
    # 样本维度
    SAMPLE_DIM = 6

    # Dimension of observation
    # 观察维度
    OBSERVATION_SHAPE = 250