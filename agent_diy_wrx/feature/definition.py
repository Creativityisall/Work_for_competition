#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2025 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


from kaiwu_agent.utils.common_func import create_cls, attached


SampleData = create_cls(
    "SampleData",               # s -> s' by action a with reward r and d'
    feature=None,               # s
    lstm_state_pi=None,         # s 时的 actor lstm 隐状态 (h,c)，训练时不翻新，off-policy
    lstm_state_vf=None,         # s 时的 critic lstm 隐状态 (h,c)，训练时不翻新，off-policy
    action=None,                # a
    logprob=None,                # pi(a|s), 训练时要算新的（advantage clip 要求 ratio 要新旧对比）
    entropy=None,               # H(·|s)，训练时要算新的（entropy loss 要新旧对比）
    advantage=None,             # 轨迹收集完毕后立刻计算，训练时不翻新，off-policy
    rreturn=None,               # 跟随 GAE 计算的折扣回报，训练时不翻新，off-policy
    value=None,                 # 采取动作转移状态之前的状态，训练时要算新的（value loss 要新旧对比）
    done=None                   # d'
) 
# model.get_action_and_value() 时，要用到（不翻新的）feature，lstm_state_pi 和 lstm_state_vf，得到翻新的 values, logprobs, entropies。 

from agent_diy.algorithm.algorithm import LstmPpoAlgorithm



############ ABANDON THESE ↓ ############
@attached
def sample_process(list_frame):
    return [SampleData(**i.__dict__) for i in list_frame]

def reward_shaping(obs_data, _obs_data, extra_info, _extra_info, terminated, truncated, frame_no, score):
    reward = 0
    return reward
############ ABANDON THESE ↑ ############



@attached
def samples_process(list_frame):
    """
    处理采样数据，将每一帧的特征转换为 SampleData 对象列表。
    """
    list_adv, list_rreturn = LstmPpoAlgorithm.compute_gae_and_rreturn(list_frame)
    
    # Note: training need advs and RETURNS, not `reward`s.
    list_sample_data = [
        SampleData(
            feature=frame.obs_data.feature,
            lstm_state_pi=frame.obs_data.lstm_state_pi,
            lstm_state_vf=frame.obs_data.lstm_state_vf,
            action=frame.act_data.action,
            logprob=frame.act_data.logprob,
            entropy=frame.act_data.entropy,
            advantage=list_adv[i],
            rreturn=list_rreturn[i],
            value=frame.act_data.value,
            done=frame.done
        )
        for i, frame in enumerate(list_frame)
    ]

    return list_sample_data 



# TODO GTL（xrq说不用搞得太复杂喧宾夺主，按照最原始最简单的来。但是我觉得也要考虑：稀疏环境的奖励，大部分时间都没有奖励函数，肯定不太好？先实现简单的再说。先读新手册。）
def reward_shaping(obs_data, _obs_data, extra_info, _extra_info, terminated, truncated, frame_no, score):

    reward = 0
    return reward