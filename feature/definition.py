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
    probs=None,
    value=None,
    target=None,
    predict=None,
    action=None,
    prob=None,
)

SampleData = create_cls("SampleData", npdata=None)


def reward_process(end_dist, history_dist, feature_vector=None, target_info=None):
    """
    改进的奖励函数设计
    
    Args:
        end_dist: 到终点的归一化距离 [0,1]
        history_dist: 到历史位置的归一化距离 [0,1] 
        feature_vector: 完整的20维特征向量 
        game_state: 游戏状态信息 
    """
    # step reward
    # 1. 基础步数惩罚 - 鼓励快速完成
    step_reward = -0.001

    # 获取动态统计的宝箱信息
    if target_info:
        target_type = target_info.get('type', 'random') #默认值是后者
        treasures_collected = target_info.get('treasures_collected', 0)
        total_treasures = target_info.get('total_treasures', 1)
        collection_progress = target_info.get('collection_progress', 0)
    else:
        target_type = 'random' #  当前Agent追求的目标类型
        treasures_collected = 0 # Agent已经收集到的宝箱数量
        total_treasures = 1 # 从游戏开始到当前步骤，Agent探索发现的宝箱总数
        collection_progress = 0 # 收集进度比例

    # 根据目标类型和收集进度设计奖励
    if target_type == 'treasure':
        # 宝箱目标：适中的奖励鼓励收集
        if end_dist < 0.05:
            treasure_reward = 0.1  # 收集宝箱奖励 (原来0.5 -> 0.1)
        elif end_dist < 0.2:
            treasure_reward = 0.02 - 0.1 * end_dist  # 适中的梯度
        else:
            treasure_reward = -0.005 * end_dist
        end_reward = treasure_reward
        
    elif target_type == 'end_point':
        # 终点目标：根据收集进度决定，但奖励更平衡
        if collection_progress >= 0.8:  # 收集了80%以上
            if end_dist < 0.05:
                end_reward = 0.3  # 完成游戏奖励 (原来1.0 -> 0.3)
            elif end_dist < 0.1:
                end_reward = 0.15 - 0.75 * end_dist  # 线性递减
            else:
                end_reward = -0.005 * end_dist
        elif collection_progress >= 0.5:  # 收集了50-80%
            if end_dist < 0.05:
                end_reward = 0.05  # 中等奖励 (原来0.2 -> 0.05)
            else:
                end_reward = -0.008 * end_dist
        else:  # 收集不足50%，适度惩罚过早到达终点
            end_reward = -0.05 * (1 - collection_progress) - 0.005 * end_dist  # 减轻惩罚
            
    elif target_type == 'buff':
        if end_dist < 0.05:
            end_reward = 0.02  # Buff奖励 (原来0.1 -> 0.02)
        else:
            end_reward = -0.002 * end_dist

    # 3. 探索奖励 - 鼓励远离历史位置
    exploration_reward = min(0.001, 0.02 * history_dist)
    
    # # 4. 进度奖励 - 基于是否找到真正的终点
    # if feature_vector is not None and len(feature_vector) >= 7:
    #     end_found = feature_vector[2]  # self.feature_end_pos[0]
    #     if end_found > 0.99:  # 找到真正的终点
    #         progress_reward = 0.01
    #     else:
    #         progress_reward = -0.005  # 轻微惩罚没找到终点
    # else:
    #     progress_reward = 0

    # 5. 卡住惩罚 - 基于合法动作数量
    
    # 6. 方向一致性奖励 - 鼓励朝目标方向移动
    direction_reward = 0
    if feature_vector is not None and len(feature_vector) >= 7:
        # feature_end_pos[1:3] 是归一化的方向向量
        direction_consistency = abs(feature_vector[3]) + abs(feature_vector[4])
        direction_reward = 0.001 * direction_consistency

    # 7. 阶段性策略奖励
    phase_reward = 0
    if collection_progress < 0.8 and target_type == 'treasure':
        # 收集阶段：额外鼓励寻找宝箱
        phase_reward = 0.005
    elif collection_progress >= 0.8 and target_type == 'end_point':
        # 完成阶段：额外鼓励到达终点
        phase_reward = 0.01

    # 8. 收集效率奖励
    efficiency_reward = 0
    if treasures_collected > 0 and total_treasures > 0:
        efficiency_reward = min(0.005, treasures_collected * 0.001)

    # 9. 历史轨迹奖励 - 鼓励保持合理的探索模式
    trajectory_reward = 0
    if 0.1 < history_dist < 0.5:  # 保持适中的探索距离
        trajectory_reward = 0.001

    # 计算总奖励
    total_reward = (
        step_reward +           # -0.001
        end_reward +            # 主要奖励 [-0.05, +0.3]
        exploration_reward +    # [0, +0.005] 探索奖励
        direction_reward +      # [0, +0.002] 方向奖励   
        phase_reward +          # [0, +0.01] 阶段奖励
        efficiency_reward +     # [0, +0.005] 效率奖励
        trajectory_reward       # [0, +0.001] 轨迹奖励
    )
    
    # 14. 奖励裁剪，防止异常值
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
        self.probs = []
        self.actions = []
        self.reward = []
        self.value = []
        self.adv = []
        self.tdlamret = []
        self.legal_action = []
        self.count = 0
        self.samples = []

    def add(self, feature, legal_action, prob, action, value, reward):
        self.feature.append(feature)
        self.legal_action.append(legal_action)
        self.probs.append(prob)
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

    def sample_process(self, feature, legal_action, prob, action, value, reward):
        self.add(feature, legal_action, prob, action, value, reward)

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
        probs = np.array(self.probs).transpose()
        actions = np.array(self.actions).transpose()
        reward = np.array(self.reward[:-1]).transpose()
        value = np.array(self.value[:-1]).transpose()
        legal_action = np.array(self.legal_action).transpose()
        adv = np.array(self.adv).transpose()
        tdlamret = np.array(self.tdlamret).transpose()

        data = np.concatenate([feature, reward, value, tdlamret, adv, actions, probs, legal_action]).transpose()

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
