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
    RewardStateTracker,
    RewardConfig
)
from kaiwu_agent.utils.common_func import Frame
from kaiwu_agent.utils.common_func import attached
from tools.train_env_conf_validate import read_usr_conf
import time
import math
import os

EPISODES = 300
REPORT_INTERVAL = 60
SAVE_INTERVAL = 300
INIT_MAX_STEPS = 2000


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
        env, agent = envs[0], agents[0]
        # Initializing monitoring data
        # 监控数据初始化
        monitor_data = {
            "reward": 0,
            "terminated": 0
        }
        last_report_monitor_time = time.time()

        logger.info("Start Training...")
        start_t = time.time()
        last_save_model_time = start_t

        total_reward, steps_cnt = 0, 0
        max_steps = INIT_MAX_STEPS
        reward_state_stack = RewardStateTracker(0)
        # 开始训练
        for episode in range(EPISODES):
            # 重置agent
            agent.reset()
            print("episode:",episode)
            time.sleep(3)
            obs, extra_info = env.reset(usr_conf=usr_conf)
            #print(obs)
            #print(extra_info)
            reward_state_stack.reset(obs["score_info"]["buff_count"])
            if extra_info["result_code"] != 0:
                logger.error(
                    f"env.reset result_code is {extra_info['result_code']}, result_message is {extra_info['result_message']}"
                )
                raise RuntimeError(extra_info["result_message"])
            done = False
            obs_data = agent.observation_process(done, obs, extra_info)
            sample_buffer = []

            # print(obs["feature"])
            # 进行采样
            for step in range(max_steps):
                steps_cnt += 1
                if done:
                    break
                # 预测动作
                list_act_data, model_version = agent.predict(list_obs_data=[obs_data])
                act_data = list_act_data[0]
                # 处理动作数据
                act = agent.action_process(act_data)
                # print(act)
                # 环境交互
                frame_no, _obs, terminated, truncated, _extra_info = env.step(act)
                # print("obs:",_obs)
                # print(_extra_info)
                # print("frame_state:",_extra_info["frame_state"]["legal_act"])
                if _extra_info["result_code"] != 0:
                    logger.error(
                        f"extra_info.result_code is {_extra_info['result_code']}, \
                        extra_info.result_message is {_extra_info['result_message']}"
                    )
                    break

                score = _obs["score_info"]["score"]
                reward = reward_shaping(reward_state_stack, frame_no, score, terminated, truncated, obs, _obs, extra_info, _extra_info,
                                        step)
                obs = _obs
                extra_info = _extra_info
                done = terminated or truncated
                # print(done)
                # 记录采样信息
                sample = Frame(
                    reward=reward,
                    done=done
                )
                sample_buffer.append(sample)
                total_reward += reward
                # if done:
                #    break
                # print("observation:",frame_no)
                # print("observation:",obs)
                # print("observation:",terminated)
                # print("observation:",truncated)
                # print("observation:",_extra_info)
                obs_data = agent.observation_process(done, obs, _extra_info)
                #print(agent.model.buffer)
            # max_steps += 100
            # 采样数据处理
            sample_data = sample_process(agent.model, sample_buffer, agent.gamma, obs, episode)
            # 学习数据
            last_state = obs_data.feature
            sample_data.last_state = last_state
            agent.learn([sample_data])
            sample_buffer = []

            # print(1)
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
            # print("OK")
            time.sleep(1)

        # eval(env, agent, logger, usr_conf)
        end_t = time.time()
        logger.info(f"Training Time for {episode + 1} episodes: {end_t - start_t} s")

        agent.save_model(id="latest")

    except Exception as e:
        raise RuntimeError(f"workflow error")
