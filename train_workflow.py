# agent_diy/train/train.py
#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
############################################################################
# Copyright © 1998 - 2025 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""
import numpy as np
import time
import os
import torch
from kaiwu_agent.utils.common_func import attached
from tools.train_env_conf_validate import read_usr_conf

# 导入新的SampleManager和修改后的数据结构
from agent_diy.feature.definition import SampleManager, RewardStateTracker, reward_shaping
from agent_diy.conf.conf import Config

@attached
def workflow(envs, agents, logger=None, monitor=None):
    """
    新的训练工作流，使用解耦的SampleManager。
    """
    try:
        env, agent = envs[0], agents[0]
        # 注意：这里的参数应从配置文件读取
        episode_num_every_epoch = Config.episodes 
        last_save_model_time = time.time()
        last_put_data_time = time.time()

        # Read and validate configuration file
        # 配置文件读取和校验
        usr_conf = read_usr_conf("agent_diy/conf/train_env_conf.toml", logger)
        if usr_conf is None:
            logger.error(f"usr_conf is None, please check agent_diy/conf/train_env_conf.toml")
            return

        while True:
            # run_episodes现在会yield一个完整的、处理好的SampleData对象
            for g_data, monitor_data in run_episodes(episode_num_every_epoch, env, agent, usr_conf, logger, monitor):
                if g_data:
                    agent.learn(g_data)

            now = time.time()
            # 定期保存模型
            if now - last_save_model_time >= 1800:
                agent.save_model(path=os.environ.get('KAIWU_AGENT_MODEL_PATH'), id='your_model_id') # 路径和ID应配置
                last_save_model_time = now

            # 定期上报监控
            if monitor and now - last_put_data_time >= 60:
                monitor.put_data({os.getpid(): monitor_data})
                last_put_data_time = now

    except Exception as e:
        logger.error(f"Workflow failed: {e}", exc_info=True)
        raise RuntimeError(f"workflow error: {e}")

def run_episodes(n_episode, env, agent, usr_conf, logger, monitor):
    """
    运行n个回合，收集数据，处理并返回。
    """
    try:
        for episode in range(n_episode):
            # 1. 在每回合开始时，创建一个新的数据收集器
            collector = SampleManager(gamma=Config.GAMMA, gae_lambda=Config.GAE_LAMDA) # GAE参数应从Config获取
            reward_tracker = RewardStateTracker(buff_count=1) # buff数量应从配置获取
            time.sleep(3)
            obs, extra_info = env.reset(usr_conf=usr_conf)
            # 错误处理
            if extra_info.get("result_code", 0) < 0:
                logger.error(f"Env reset failed: {extra_info.get('result_message')}")
                continue

            agent.reset()
            agent.load_model(id="latest")

            done, truncated, terminated = False, False, False
            step = 0
            
            while not done:
                # 2. 特征处理
                obs_data = agent.observation_process(done, obs, extra_info)

                # 3. Agent进行推理，获取预测所需的所有信息
                # model.predict现在返回 (action, log_prob, value, state_seq, action_seq)
                pred_outputs = agent.predict([obs_data])
                #print(pred_outputs[0][0])
                action_tensor, log_prob, value, state_seq, action_seq = pred_outputs[0][0]
                act = agent.action_process(action_tensor)
                
                # 4. 与环境交互
                step_no, _obs, _terminated, _truncated, _extra_info = env.step(act)
                
                # 更新状态
                terminated, truncated = _terminated, _truncated
                done = terminated or truncated

                # 5. 计算奖励
                reward = reward_shaping(reward_tracker, step_no, terminated, truncated, obs, _obs, extra_info, _extra_info, step)
                
                # 6. 将所有数据显式地添加到收集器
                collector.add(
                    state_seq=state_seq,
                    action_seq=action_seq,
                    action=action_tensor,
                    logprob=log_prob,
                    value=value,
                    reward=reward,
                    done=float(done) # 将bool转为float
                )

                # 更新状态和步数
                obs, extra_info = _obs, _extra_info
                step += 1

            # 7. 回合结束，处理最后一帧并计算GAE
            if done:
                # 获取最后一帧的状态价值，用于GAE计算
                with torch.no_grad():
                    last_obs_data = agent.observation_process(done, obs, extra_info)
                    # 调用模型获取价值，但不收集数据
                    _, _, last_value, _, _ = agent.model.predict(last_obs_data.feature, done, last_obs_data.legal_actions)
                    last_value = last_value.cpu().item()

                # 触发GAE和Returns的计算
                collector.process_last_frame(last_value, float(done))

                # 获取打包好的数据
                sample_data = collector.get_data()

                # 监控数据
                monitor_data = {"total_score": extra_info["game_info"]["total_score"]}
                
                # 使用yield返回数据
                yield sample_data, monitor_data

    except Exception as e:
        logger.error(f"Run_episodes failed: {e}", exc_info=True)
        raise RuntimeError(f"run_episodes error: {e}")
