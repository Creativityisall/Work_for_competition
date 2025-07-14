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
    EPSILON = 0.2
    K_STEPS = 10
    LR_ACTOR = 3e-4
    LR_CRITIC = 3e-4
    LR_LSTM = 3e-4
    LOSS_WEIGHT = {'actor': 1.0, 'critic': 0.5, 'entropy': 0.05}