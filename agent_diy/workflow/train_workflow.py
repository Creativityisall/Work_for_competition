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
from tools.metrics_utils import get_training_metrics
import time
import math
import os

EPISODES = 10
REPORT_INTERVAL = 60
SAVE_INTERVAL = 300
INIT_MAX_STEPS = 1000

@attached
def workflow(envs, agents, logger=None, monitor=None):
    """
    Users can define their own training workflows here
    用户可以在此处自行定义训练工作流
    """

    try:
        print("v4")
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
            "reward": 0,
        }
        last_report_monitor_time = time.time()

        logger.info("Start Training...")
        start_t = time.time()
        last_save_model_time = start_t

        total_reward, steps_cnt = 0, 0
        max_steps = INIT_MAX_STEPS
        # 开始训练
        for episode in range(EPISODES):
            # Retrieving training metrics
            # 获取训练中的指标
            training_metrics = get_training_metrics()
            if training_metrics:
                logger.info(f"training_metrics is {training_metrics}")
            # Reset the game and get the initial state
            # 重置游戏, 并获取初始状态
            obs, extra_info = env.reset(usr_conf=usr_conf)
            if extra_info["result_code"] != 0:
                logger.error(
                    f"env.reset result_code is {extra_info['result_code']}, result_message is {extra_info['result_message']}"
                )
                raise RuntimeError(extra_info["result_message"])
            
            obs_data = agent.observation_process(obs, extra_info)
            sample_buffer = []
            done = False
            agent.reset()
            # 进行采样
            for step in range(max_steps):
                steps_cnt += 1
                # 预测动作
                list_act_data, model_version = agent.predict(list_obs_data=[obs_data])
                act_data = list_act_data[0]
                # 处理动作数据
                act = agent.action_process(act_data)
                # 环境交互
                frame_no, next_obs, terminated, truncated, next_extra_info = env.step(act)
                if next_extra_info["result_code"] != 0:
                    logger.error(
                        f"extra_info.result_code is {_extra_info['result_code']}, \
                        extra_info.result_message is {_extra_info['result_message']}"
                    )
                    break
                
                reward = reward_shaping(frame_no, terminated, truncated, obs, next_obs, extra_info, next_extra_info, step)
                obs = next_obs
                extra_info = next_extra_info
                done = terminated or truncated
                # logger.info(f"act {act}, reward {reward}, terminated {terminated}, truncated {truncated}")
                # 记录采样信息
                sample = Frame(
                    reward=reward,
                    done=done
                )
                sample_buffer.append(sample)
                total_reward += reward

                obs_data = agent.observation_process(obs, extra_info)
                if done:
                    # 做最后一次预测但不做处理，储存结束时刻的state
                    list_act_data, model_version = agent.predict(list_obs_data=[obs_data])
                    break
        
            max_steps = min(max_steps + 100, 2000)
            import numpy as np
            logger.info([np.exp(prob.item()) for prob in agent.model.buffer['logprobs']])
            # 采样数据处理
            sample_data = sample_process(sample_buffer)
            # 学习数据
            agent.learn(sample_data)


            now = time.time()
            # 记录参数
            if now - last_report_monitor_time > REPORT_INTERVAL:
                avg_reward = total_reward / steps_cnt
                monitor_data["reward"] = avg_reward
                if monitor:
                    monitor.put_data({os.getpid(): monitor_data})
                total_reward = 0
                steps_cnt = 0
                last_report_monitor_time = now
            # 保存模型
            if now - last_save_model_time > SAVE_INTERVAL:
                agent.save_model(id=str(episode + 1))
                last_save_model_time = now

        agent.save_model(id="latest")
        end_t = time.time()
        logger.info(f"Training Time for {EPISODES} episodes: {end_t - start_t} s")

    except Exception as e:
        raise RuntimeError(f"workflow error")