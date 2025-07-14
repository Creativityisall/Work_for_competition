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
    MINIBATCH = 256

    LSTM_SEQ_LENGTH = 128
    INPUT_SIZE = 88
    OUTPUT_SIZE = 4
    LSTM_HIDDEN_SIZE = 128
    LSTM_HIDDEN_LAYERS = 2

    GAMMA = 0.95
    LAM = 0.95
    EPSILON = 0.1
    K_STEPS = 10
    LR_PPO = 1e-5
    LR_SCHEDULER = 0.1
    LOSS_WEIGHT = {'policy': 1.0, 'value': 0.5, 'entropy': 0.015}

    SCHEDULER_STEP = 10
    SCHEDULER_LR = 0.1
