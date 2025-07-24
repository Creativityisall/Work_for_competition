#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2025 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


from agent_diy.feature.definition import (
    sample_process,
    reward_shaping,
)
from kaiwu_agent.utils.common_func import Frame
from kaiwu_agent.utils.common_func import attached
from tools.train_env_conf_validate import read_usr_conf
import time
import math
import os

from agent_diy.agent import Agent
import torch


@attached
def workflow(envs, agents, logger=None, monitor=None):
    """
    Users can define their own training workflows here
    用户可以在此处自行定义训练工作流
    """

    try:
        # Read and validate configuration file
        # 配置文件读取和校验
        usr_conf = read_usr_conf("agent_diy/conf/train_env_conf.toml", logger)
        if usr_conf is None:
            logger.error("usr_conf is None, please check agent_diy/conf/train_env_conf.toml")
            return
        
        # # Initializing monitoring data
        # # 监控数据初始化
        # monitor_data = {
        #     "reward": 0,
        #     "diy_1": 0,
        #     "diy_2": 0,
        #     "diy_3": 0,
        #     "diy_4": 0,
        #     "diy_5": 0,
        # }
        # last_report_monitor_time = time.time()

        # logger.info("Start Training...")
        start_t = time.time()
        last_save_model_time = start_t


        # Training loop
        env = envs[0]
        agent : Agent = agents[0]
        num_epochs = 1000
        num_episodes_per_epoch = 1000

        for epoch in range(num_epochs):
            for g_data in run_episode(num_episodes_per_epoch, env, agent, logger, monitor, usr_conf):
                agent.learn(g_data)
                g_data.clear()

            now = time.time()
            if now - last_save_model_time > 300:
                agent.save_model()
                last_save_model_time = now
            



    except Exception as e:
        raise RuntimeError(f"workflow error")


def run_episode(n_episodes, env, agent : Agent, loggger=None, monitor=None, usr_conf=None):
    for episode in range(n_episodes):
        done = False
        collector = list()
        
        obs, extra_info = env.reset(usr_conf=usr_conf)
        agent.load_model(id="latest")

        while not done:
            obs_data = agent.observation_process(obs, extra_info) # raw data -> (feature_dim, )

            act_data = agent.predict(list_obs_data=[obs_data])[0] # list_obs_data and [0] is for KOG 3v3, here we only need 1 obs_data to predict 1 (the first) act_data 

            # act = agent.action_process(act_data)
            act = act_data.act

            frame_no, _obs, score, terminated, truncated, _extra_info = env.step(act)
            if _obs == None:
                raise RuntimeError("env.step return None obs")
            
            
            _obs_data = agent.observation_process(_obs, _extra_info)

            # Disaster recovery
            # Give up current collector and straight to next episode 
            if truncated and frame_no:
                break

            reward = reward_shaping(obs_data, _obs_data, extra_info, _extra_info, terminated, truncated, frame_no, score)

            done = terminated or truncated

            # Construct sample
            frame = Frame(
                obs=obs_data.feature,
                _obs=_obs_data.feature,
                act=act,
                r=reward,
                done=done,
            )

            sample = sample_process([frame])[0]

            collector.append(sample)


            if done:
                if len(collector) > 0:
                    collector = sample_process(collector)
                    yield collector

                break

            obs_data = _obs_data
            obs = _obs
            extra_info = _extra_info

