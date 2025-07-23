#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2025 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


# Configuration, including dimension settings, algorithm parameter settings.
# The last few configurations in the file are for the Kaiwu platform to use and should not be changed.
# 配置，包含维度设置，算法参数设置，文件的最后一些配置是开悟平台使用不要改动
class Config:

    # Input dimension of reverb sample on learner. Note that different algorithms have different dimensions.
    # For example, the dimension of dqn in the sample code is 21624, and the dimension of target_dqn is 21624
    # **Note**, this item must be configured correctly and should be aligned with the NumpyData2SampleData function data in definition.py
    # Otherwise the sample dimension error may be reported
    # learner上reverb样本的输入维度, 注意不同的算法维度不一样, 比如示例代码中dqn的维度是21624, target_dqn的维度是21624
    # **注意**，此项必须正确配置，应该与definition.py中的NumpyData2SampleData函数数据对齐，否则可能报样本维度错误
    SAMPLE_DIM = 21624

    # Size of observation. After users design their own features, they should set the correct dimensions
    # observation的维度，用户设计了自己的特征之后应该设置正确的维度
    DIM_OF_OBSERVATION = 0

    # Dimension of movement action direction
    # 移动动作方向的维度
    DIM_OF_ACTION_DIRECTION = 8

    # Dimension of flash action direction
    # 闪现动作方向的维度
    DIM_OF_TALENT = 8
    LSTM_SEQ_LENGTH = 64
    INPUT_SIZE = 486
    OUTPUT_SIZE = 16
    LSTM_HIDDEN_SIZE = 256
    LSTM_HIDDEN_LAYERS = 1

    GAMMA = 0.95
    GAE_LAMDA = 0.95
    EPSILON = 0.1
    K_STEPS = 6
    LR_ACTOR = 1e-4
    LR_CRITIC = 1e-4
    LR_LSTM = 1e-4
    LOSS_WEIGHT = {'actor': 0.6, 'critic': 0.6, 'entropy': 0.015}#0.015

    # dimensionality of the sample
    # 样本维度
    SAMPLE_DIM = 1

    # Dimension of movement action direction
    # 移动动作方向的维度
    OBSERVATION_SHAPE = 214
    episodes = 300

    # Initial learning rate
    # 初始的学习率
    START_LR = 0.0003

    # entropy regularization coefficient
    # 熵正则化系数
    BETA_START = 0.0005

    # clip parameter
    # 裁剪参数
    CLIP_PARAM = 0.2

    # value function loss coefficient
    # 价值函数损失的系数
    VF_COEF = 1

    # actions
    # 动作
    ACTION_LEN = 1
    ACTION_NUM = 8

    # features
    # 特征
    FEATURES = [
        2,
        6,
        6,
        8,
    ]

    FEATURE_SPLIT_SHAPE = FEATURES
    FEATURE_LEN = sum(FEATURE_SPLIT_SHAPE)
    VALUE_NUM = 1
    DATA_SPLIT_SHAPE = [
        FEATURE_LEN,
        VALUE_NUM,
        VALUE_NUM,
        VALUE_NUM,
        VALUE_NUM,
        ACTION_LEN,
        ACTION_LEN,
        ACTION_NUM,
    ]
    data_len = sum(DATA_SPLIT_SHAPE)

    # Input dimension of reverb sample on learner. Note that different algorithms have different dimensions.
    # **Note**, this item must be configured correctly and should be aligned with the NumpyData2SampleData function data in definition.py
    # Otherwise the sample dimension error may be reported
    # learner上reverb样本的输入维度
    # **注意**，此项必须正确配置，应该与definition.py中的NumpyData2SampleData函数数据对齐，否则可能报样本维度错误
    SAMPLE_DIM = data_len
