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

import os
import time
import numpy as np

EPOCHES = 1000
EPISODES = 3
REPORT_INTERVAL = 60
SAVE_INTERVAL = 300
INIT_MAX_STEPS = 1000
STEPS_INTERVAL = 100

class GameBuffer:
    def __init__(self, episode=0):
        self.game_episode = episode
        # (n_steps, n_envs, dim)
        self.features = []
        self.next_features = []
        self.actions = []
        self.rewards = []
        self.dones = []

    def __getitem__(self, key):
        return self.__dict__.get(key, None)
    
    def add(self, frame, otherdata=None):
        self.features.append(frame.list_obs_data)
        self.next_features.append(frame.list_next_obs_data)
        self.actions.append(frame.actions)
        self.rewards.append(frame.rewards)
        self.dones.append(frame.dones)

        for k, v in otherdata.items():
            if k not in self.__dict__:
                self.__dict__[k] = []
            self.__dict__[k].append(v)

    def compute_returns_and_advantage(self, agent, gamma=0.99, gae_lambda=0.95):
        """计算返回和优势"""
        values = agent.compute_values(np.array(self.features), self.lstm_states) # (n_steps, n_envs, )
        n_steps, n_envs = values.shape
        next_values = np.zeros(n_envs, dtype=np.float32)
        
        self.advantages = np.zeros((n_steps, n_envs), dtype=np.float32)
        self.returns = np.zeros((n_steps, n_envs), dtype=np.float32)
        rewards = np.array(self.rewards, dtype=np.float32)
        dones = np.array(self.dones, dtype=np.float32)
        for step in reversed(range(n_steps)):
            delta = rewards[step] + gamma * next_values * (1.0 - dones[step]) - values[step]
            self.advantages[step] = delta + gamma * gae_lambda * (1.0 - dones[step]) * \
                    (self.advantages[step + 1] if step + 1 < n_steps else 0.0)
            self.returns[step] = self.advantages[step] + values[step]
            next_values = values[step] * (1.0 - dones[step])

        self.advantages = (self.advantages - self.advantages.mean(axis=0)) / (self.advantages.std(axis=0) + 1e-6)
        self.returns = (self.returns - self.returns.mean(axis=0)) / (self.returns.std(axis=0) + 1e-6)
        
    def get_monitor_data(self):
        monitor_data = {
            "reward": np.sum(self.rewards) / len(self.rewards) if self.rewards else 0,
        }
        return monitor_data

    def clear(self):
        self.features.clear()
        self.next_features.clear()
        self.actions.clear()
        self.rewards.clear()
        self.dones.clear()


@attached
def workflow(envs, agents, logger=None, monitor=None):
    try:
        # ------------------------------------- Initialization ------------------------------------- #
        # 配置文件读取和校验
        usr_conf = read_usr_conf("agent_diy/conf/train_env_conf.toml", logger)
        if usr_conf is None:
            logger.error("usr_conf is None, please check agent_diy/conf/train_env_conf.toml")
            return
        
        env, agent = envs[0], agents[0]
        # 监控数据初始化
        monitor_data = {}

        logger.info("Start Training...")
        start_t = time.time()
        last_save_model_time = start_t
        # ------------------------------------- Initialization ------------------------------------- #

        # ------------------------------------- Training Loop ------------------------------------- #
        max_steps = INIT_MAX_STEPS
        for epoch in range(EPOCHES):
            for sample in run_episode(agent, env, usr_conf, max_steps, logger, monitor):
                agent.learn(sample)
                if monitor:
                    # 监控DIY数据
                    monitor_data = {"diy_1": 0.0, "diy_2": 0.0, "diy_3": 0.0, "diy_4": 0.0, "diy_5": 0.0}
                    otherdata = agent.get_other_monitor_data()
                    for i, item in enumerate(otherdata.items()):
                        if i > 5: 
                            break
                        k, v = item
                        monitor_data[f"diy_{i}"] = v

                    monitor.put_data({os.getpid(): monitor_data})

            max_steps = min(max_steps + STEPS_INTERVAL, 2000)  # 平滑增加步数
            # 阶段性保存
            now = time.time()
            if now - last_save_model_time > SAVE_INTERVAL:
                agent.save_model(id=epoch + 1)
                last_save_model_time = now
        # ------------------------------------- Training Loop ------------------------------------- #     
        logger.info("Train Over")
        time.sleep(30) # 等待保存
        agent.save_model(id="latest")
        end_t = time.time()
        logger.info(f"Training Time for {EPISODES} episodes: {end_t - start_t} s")
        
    except Exception as e:
        raise RuntimeError(f"workflow error: {e}")

def run_episode(agent, env, usr_conf, max_steps, logger, monitor):
    for episode in range(EPISODES):
        # 重置agent与env
        agent.reset()
        obs, extra_info = env.reset(usr_conf=usr_conf)
        if extra_info["result_code"] != 0:
            logger.error(
                f"env.reset result_code is {extra_info['result_code']}, result_message is {extra_info['result_message']}"
            )
            raise RuntimeError(extra_info["result_message"])
        
        list_obs_data = agent.observation_process(list_obs=[obs], list_extra_info=[extra_info])
        # 记录游戏数据
        game_collector = GameBuffer(episode=episode)
        # 开始游戏
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
                raise RuntimeError(next_extra_info["result_message"])
            
            # 奖励塑形
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
            # 超时数据处理
            rewards = agent.handle_timeout(truncateds=[truncated], rewards=rewards, list_obs_data=list_obs_data)

            # next obs
            list_next_obs_data = agent.observation_process(list_obs=[next_obs], list_extra_info=[next_extra_info])

            # 采样训练用游戏内数据
            otherdata = agent.get_other_sample_data()
            dones = [terminated or truncated]
            frame = Frame(
                list_obs_data = list_obs_data,              # S     /   list[ObsData]
                list_next_obs_data = list_next_obs_data,    # S'    /   list[ObsData]
                actions = actions,                          # A     /   list[int]                           
                rewards = rewards,                          # R     /   list[float]     
                dones = dones,                              # D     /   list[bool]  
            )
            # 收集游戏数据
            game_collector.add(frame, otherdata=otherdata)

            if dones[0]: # TODO: 分布式
                break

            # 更新状态
            obs, extra_info = next_obs, next_extra_info
            list_obs_data = list_next_obs_data
            

        if monitor:
            monitor_data = game_collector.get_monitor_data()
            monitor.put_data({os.getpid(): monitor_data})

        # 计算返回和优势
        game_collector.compute_returns_and_advantage(agent, gamma=None, gae_lambda=None)
        # 收集采样数据
        sample = sample_process(game_collector) # (n_steps, n_envs, dim) -> (n_steps * n_envs, dim)
        yield sample 

# MODEL_ID = 300
# @attached
# def workflow(envs, agents, logger=None, monitor=None): # for Test
#     max_score = 0
#     usr_conf = read_usr_conf("agent_diy/conf/train_env_conf.toml", logger)
#     if usr_conf is None:
#         logger.error("usr_conf is None, please check agent_diy/conf/train_env_conf.toml")
#         return
#     env, agent = envs[0], agents[0]
#     logger.info("Start Testing...")
#     agent.reset()

#     agent.load_model(path="agent_diy", id=MODEL_ID)
#     obs, extra_info = env.reset(usr_conf=usr_conf)

#     list_obs_data = agent.observation_process(list_obs=[obs], list_extra_info=[extra_info])
#     agent.set_feature(list_obs_data)
#     for step in range(2000):
#         action, model_version = agent.exploit(list_obs_data={"obs": obs, "extra_info": extra_info})

#         frame_no, next_obs, terminated, truncated, next_extra_info = env.step(action)
#         if next_extra_info["result_code"] != 0:
#             logger.error(
#                 f"extra_info.result_code is {next_extra_info['result_code']}, \
#                 extra_info.result_message is {next_extra_info['result_message']}"
#             )
#             break
        
#         rewards = reward_shaping(
#             list_frame_no=[frame_no], 
#             list_terminated=[terminated], 
#             list_truncated=[truncated], 
#             list_obs=[obs], 
#             list_next_obs=[next_obs], 
#             list_extra_info=[extra_info], 
#             list_next_extra_info=[next_extra_info], 
#             step=step
#         ) # (n_envs, )

#         obs, extra_info = next_obs, next_extra_info

#         dones = [terminated or truncated]

#         score = extra_info['game_info']['score']
#         max_score = max(max_score, score)
#         total_score = extra_info['game_info']['total_score']
#         logger.info(f"reward {rewards[0]}, score {score}")
#         if dones[0]:
#             logger.info(f"---- Test Over, Total {step} steps ----")
#             logger.info(f"Total Score {total_score}")
#             return
        
#     logger.info(f"---- Test Over, Time Over ----")
#     logger.info(f"Max Score {max_score}")
