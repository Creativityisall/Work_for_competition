import time
import os

from utils import create_cls
from feature import (
    sample_process, 
    reward_shaping
)
from logger_monitor import init_logger, init_monitor
from agent import Agent
from env import Env

Frame = create_cls("Frame", rewards=None, dones=None)

EPISODES = 300
REPORT_INTERVAL = 60
SAVE_INTERVAL = 300
INIT_MAX_STEPS = 1000
STEPS_INTERVAL = 100

def workflow(envs, agents, logger=None, monitor=None):
    """
    Users can define their own training workflows here
    用户可以在此处自行定义训练工作流
    """
    try:
        # 配置文件读取和校验
        env, agent = envs[0], agents[0]
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
            obs, extra_info = env.reset() # TODO: 分布式
            if extra_info["result_code"] != 0:
                logger.error(
                    f"env.reset result_code is {extra_info['result_code']}, result_message is {extra_info['result_message']}"
                )
                raise RuntimeError(extra_info["result_message"])
            list_obs_data = agent.observation_process(list_obs=[obs], list_extra_info=[extra_info])
            agent.set_feature(list_obs_data)
            for step in range(max_steps):
                # 预测动作
                list_act_data = agent.predict(list_obs_data=list_obs_data)
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
                agent.learn()
            now = time.time()
            # 记录参数
            if now - last_report_monitor_time > REPORT_INTERVAL:
                if monitor:
                    monitor_data['reward'] = 100 * monitor_data['reward'] / max_steps
                    logger.info(f"reward {monitor_data['reward']}")
                    monitor.put_data(monitor_data)
                    monitor_data['reward'] = 0

                last_report_monitor_time = now
            # 保存模型
            if now - last_save_model_time > SAVE_INTERVAL:
                agent.save_model(path="./Test/backup", id=episode+1)
                last_save_model_time = now

            # 更平缓地增加步数
            if episode % 5 == 0:
                max_steps = min(max_steps + 50, 2000)

        logger.info("Train Over")
        time.sleep(30) # 等待保存
        agent.save_model(path="./Test/backup", id="latest")
        end_t = time.time()
        logger.info(f"Training Time for {EPISODES} episodes: {end_t - start_t} s")

    except Exception as e:
        raise RuntimeError(f"workflow error")
    

if __name__ == "__main__":
    logger = init_logger(name="test")
    monitor = init_monitor()
    envs = [Env()]
    agents = [Agent(agent_type="player", device="cpu", logger=logger, monitor=monitor)]
    workflow(envs, agents, logger, monitor)
    monitor.draw()