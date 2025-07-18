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
from kaiwu_agent.utils.common_func import attached, create_cls
from tools.train_env_conf_validate import read_usr_conf
from tools.metrics_utils import get_training_metrics
import time
import numpy as np
import os

EPISODES = 300
REPORT_INTERVAL = 60
SAVE_INTERVAL = 300
INIT_MAX_STEPS = 1000
STEPS_INTERVAL = 100

placeholder = [] # 迫于官方架构的装饰器限制，故添加

@attached
def workflow(envs, agents, logger=None, monitor=None):
    """
    Users can define their own training workflows here
    用户可以在此处自行定义训练工作流
    """
    try:
        print("v8")
        # Read and validate configuration file
        # 配置文件读取和校验
        usr_conf = read_usr_conf("agent_diy/conf/train_env_conf.toml", logger)
        if usr_conf is None:
            logger.error("usr_conf is None, please check agent_diy/conf/train_env_conf.toml")
            return

        env, agent = envs[0], agents[0]
        
        # Initializing monitoring data
        # 监控数据初始化
        monitor_data = {
            "reward": 0.0,
        }

        last_report_monitor_time = time.time()

        logger.info("Start Training...")
        start_t = time.time()
        last_save_model_time = start_t

        max_steps = INIT_MAX_STEPS
        # 开始训练
        for episode in range(EPISODES):
            agent.reset()
            # 重置游戏, 并获取初始状态
            time.sleep(3) # TODO: 删了报错，我不理解的bug?与env.step/reset 有关
            obs, extra_info = env.reset(usr_conf=usr_conf) # TODO: 分布式
            if extra_info["result_code"] != 0:
                logger.error(
                    f"env.reset result_code is {extra_info['result_code']}, result_message is {extra_info['result_message']}"
                )
                raise RuntimeError(extra_info["result_message"])
            list_obs_data = agent.observation_process(list_obs=[obs], list_extra_info=[extra_info])
            agent.set_feature(list_obs_data)
            for step in range(max_steps):
                # 预测动作
                list_act_data, model_version = agent.predict(list_obs_data=list_obs_data)
                # 处理动作数据
                actions = agent.action_process(list_act_data)
                # 环境交互
                action = actions[0] # 没有分布式，故动作列表只有一个

                frame_no, next_obs, terminated, truncated, next_extra_info = env.step(action)
                if next_extra_info["result_code"] != 0:
                    logger.error(
                        f"extra_info.result_code is {next_extra_info['result_code']}, \
                        extra_info.result_message is {next_extra_info['result_message']}"
                    )
                    break
                
                rewards = reward_shaping(
                    list_frame_no=[frame_no], 
                    list_terminated=[terminated], 
                    list_truncated=[truncated], 
                    list_obs=[obs], 
                    list_next_obs=[next_obs], 
                    list_extra_info=[extra_info], 
                    list_next_extra_info=[next_extra_info], 
                    step=step
                ) # (n_envs, )

                obs, extra_info = next_obs, next_extra_info
                # 没有分布式，故环境列表只有一个
                # dones = np.logical_or(terminated, truncated)
                dones = [terminated or truncated]
                # 超时数据处理
                rewards = agent.handle_timeout(truncateds=[truncated], rewards=rewards, list_obs_data=list_obs_data)
                # 采样
                sample = Frame(
                    rewards=rewards,
                    dones=dones
                )
                sample_data = sample_process(sample)
                # next obs
                list_obs_data = agent.observation_process(list_obs=[obs], list_extra_info=[extra_info])
                # 收集采样数据
                agent.collect(sample_data, list_obs_data)
                # 记录参数
                # logger.info(f"prob - reward: {list_act_data[0].prob} - {rewards[0]}")
                monitor_data['reward'] += rewards[0]

                if dones[0]: # TODO: 分布式
                    break

            agent.compute_returns_and_advantage()
            # 学习数据
            if agent.collect_full(): # 如果buffer填充满，则开始学习
                logger.info(f" ---------- {episode} Start Learn ----------- ")
                agent.learn(placeholder)
            now = time.time()
            # 记录参数
            if now - last_report_monitor_time > REPORT_INTERVAL:
                if monitor:
                    monitor_data['reward'] = 100 * monitor_data['reward'] / max_steps
                    logger.info(f"reward {monitor_data['reward']}")
                    monitor.put_data({os.getpid(): monitor_data})
                    monitor_data['reward'] = 0

                last_report_monitor_time = now
            # 保存模型
            if now - last_save_model_time > SAVE_INTERVAL:
                agent.save_model(id=episode+1)
                last_save_model_time = now

            # 更平缓地增加步数
            if episode % 5 == 0:
                max_steps = min(max_steps + 50, 2000)

        logger.info("Train Over")
        time.sleep(30) # 等待保存
        agent.save_model(id="latest")
        end_t = time.time()
        logger.info(f"Training Time for {EPISODES} episodes: {end_t - start_t} s")

    except Exception as e:
        raise RuntimeError(f"workflow error")

