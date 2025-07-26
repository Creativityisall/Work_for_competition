#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2025 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""

from agent_ppo.conf.conf import Config
from kaiwu_agent.utils.common_func import create_cls, attached
import numpy as np

# The create_cls function is used to dynamically create a class.
# The first parameter of the function is the type name, and the remaining parameters are the attributes of the class.
# The default value of the attribute should be set to None.
# create_cls函数用于动态创建一个类，函数第一个参数为类型名称，剩余参数为类的属性，属性默认值应设为None
ObsData = create_cls(
    "ObsData",
    feature=None,
    legal_action=None,
    reward=None,
)


ActData = create_cls(
    "ActData",
    log_probs=None,
    value=None,
    target=None,
    predict=None,
    action=None,
    log_prob=None,
)

SampleData = create_cls("SampleData", npdata=None)


# TODO better reward design ??? Keep it simple.

def reward_process(
    step_no,
    cur_pos,
    cur_pos_norm,

    undetected_area,
    cnt_new_detected,
    
    treasure_buf_pos_list,
    destination_pos,
    has_found_dest,

    legal_action,
    talent_available,
    talent_cd
):
    """
    改进的奖励函数设计
    
    Args:
        cur_pos: 智能体当前坐标。
        map: 智能体存储的全局地图探索情况：-1 表示未探索，0表示已探索，1表示障碍物。
        treasure_pos_list: 智能体存储的宝藏位置列表，包含所有宝藏的坐标。
        destination_pos: 智能体存储的目标位置坐标（没找到则为 None）。
    """
    # step reward
    # 1. 基础步数惩罚 - 鼓励快速完成
    step_reward = -0.001


    # 2. 探索奖励 - 鼓励探索未探索区域    
    # ...

    # 计算总奖励
    total_reward = 0
    total_reward = max(-0.2, min(0.5, total_reward))

    return [total_reward]


class SampleManager:
    def __init__(
        self,
        gamma=0.99,
        tdlambda=0.95,
    ):
        self.gamma = Config.GAMMA
        self.tdlambda = Config.TDLAMBDA

        self.feature = []
        self.log_probs = []
        self.actions = []
        self.reward = []
        self.value = []
        self.adv = []
        self.tdlamret = []
        self.legal_action = []
        self.count = 0
        self.samples = []

    def add(self, feature, legal_action, log_prob, action, value, reward):
        self.feature.append(feature)
        self.legal_action.append(legal_action)
        self.log_probs.append(log_prob)
        self.actions.append(action)
        self.value.append(value)
        self.reward.append(reward)
        self.adv.append(np.zeros_like(value))
        self.tdlamret.append(np.zeros_like(value))
        self.count += 1

    def add_last_reward(self, reward):
        self.reward.append(reward)
        self.value.append(np.zeros_like(reward))

    def update_sample_info(self):
        last_gae = 0
        for i in range(self.count - 1, -1, -1):
            reward = self.reward[i + 1]
            next_val = self.value[i + 1]
            val = self.value[i]
            delta = reward + next_val * self.gamma - val
            last_gae = delta + self.gamma * self.tdlambda * last_gae
            self.adv[i] = last_gae
            self.tdlamret[i] = last_gae + val

    def sample_process(self, feature, legal_action, log_prob, action, value, reward):
        self.add(feature, legal_action, log_prob, action, value, reward)

    def process_last_frame(self, reward):
        self.add_last_reward(reward)
        # 发送前的后向传递更新
        # Backward pass updates before sending
        self.update_sample_info()
        self.samples = self._get_game_data()

    def get_game_data(self):
        ret = self.samples
        self.samples = []
        return ret

    def _get_game_data(self):
        feature = np.array(self.feature).transpose()
        log_probs = np.array(self.log_probs).transpose()
        actions = np.array(self.actions).transpose()
        reward = np.array(self.reward[:-1]).transpose()
        value = np.array(self.value[:-1]).transpose()
        legal_action = np.array(self.legal_action).transpose()
        adv = np.array(self.adv).transpose()
        tdlamret = np.array(self.tdlamret).transpose()

        data = np.concatenate([feature, reward, value, tdlamret, adv, actions, log_probs, legal_action]).transpose()

        samples = []
        for i in range(0, self.count):
            samples.append(SampleData(npdata=data[i].astype(np.float32)))

        return samples


@attached
def SampleData2NumpyData(g_data):
    return g_data.npdata


@attached
def NumpyData2SampleData(s_data):
    return SampleData(npdata=s_data)
