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
    LSTM_SEQ_LENGTH = 64
    INPUT_SIZE = 88
    OUTPUT_SIZE = 4
    LSTM_HIDDEN_SIZE = 64
    LSTM_HIDDEN_LAYERS = 1

    GAMMA = 0.95
    EPSILON = 0.1
    K_STEPS = 6
    LR_ACTOR = 1e-4
    LR_CRITIC = 1e-4
    LR_LSTM = 1e-4
    LOSS_WEIGHT = {'actor': 0.6, 'critic': 0.6, 'entropy': 1}#0.015

    # dimensionality of the sample
    # 样本维度
    SAMPLE_DIM = 1

    # Dimension of movement action direction
    # 移动动作方向的维度
    OBSERVATION_SHAPE = 214
    episodes = 300
