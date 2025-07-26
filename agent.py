#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2025 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


import torch
from kaiwu_agent.agent.base_agent import (
    predict_wrapper,
    exploit_wrapper,
    learn_wrapper,
    save_model_wrapper,
    load_model_wrapper,
    BaseAgent,
)

import random
import numpy as np
from kaiwu_agent.utils.common_func import attached
from agent_ppo.model.model import NetworkModelActor
from agent_ppo.algorithm.algorithm import Algorithm
from agent_ppo.feature.definition import SampleData, ObsData, ActData, SampleManager
from agent_ppo.feature.preprocessor import Preprocessor

###############################################
import inspect
import numpy as np
import torch
from collections.abc import Iterable  # 兼容更多可迭代对象
from functools import wraps

def my_trace(func):
    """装饰器：显示参数名、返回值名及深度类型信息"""
    @wraps(func)  # 保留原函数元信息
    def wrapper(*args, **kwargs):
        func_name = func.__name__
        sig = inspect.signature(func)
        bound_args = sig.bind(*args, **kwargs)
        bound_args.apply_defaults()  # 填充默认参数

        # 打印输入信息（带参数名）
        print(f"\n[TRACE] 函数 {func_name}() 输入:")
        for name, value in bound_args.arguments.items():
            print(f"  - 参数 {name}: {_get_deep_arg_info(value, indent=4)}")

        # 执行函数并捕获返回值
        result = func(*args, **kwargs)

        # 打印输出信息（尝试获取返回值变量名）
        result_name = _get_return_var_name(func)
        print(f"[TRACE] 函数 {func_name}() 输出 {'-> ' + result_name + ': ' if result_name else ''}{_get_deep_arg_info(result)}")
        return result
    return wrapper

def _get_deep_arg_info(arg, indent=0, max_elements=5):
    """递归获取变量及其子元素的类型和结构信息"""
    indent_str = ' ' * indent
    info = []
    
    # 处理数组/张量
    if isinstance(arg, np.ndarray):
        info.append(f"ndarray shape={arg.shape}")
    elif isinstance(arg, torch.Tensor):
        info.append(f"Tensor shape={arg.shape} (dtype={arg.dtype}, device={arg.device})")
    
    # 处理列表/元组（递归检查元素）
    elif isinstance(arg, (list, tuple)):
        info.append(f"{type(arg).__name__} len={len(arg)} [")
        for i, item in enumerate(arg[:max_elements]):  # 限制打印的元素数量
            item_info = _get_deep_arg_info(item, indent + 4)
            info.append(f"{indent_str}  [{i}]: {item_info}")
        if len(arg) > max_elements:
            info.append(f"{indent_str}  ... (仅显示前 {max_elements}/{len(arg)} 个元素)")
        info.append(f"{indent_str}]")
    
    # 处理字典（递归检查值）
    elif isinstance(arg, dict):
        info.append(f"dict len={len(arg)} {{")
        for k, v in list(arg.items())[:max_elements]:
            item_info = _get_deep_arg_info(v, indent + 4)
            info.append(f"{indent_str}  '{k}': {item_info}")
        if len(arg) > max_elements:
            info.append(f"{indent_str}  ... (仅显示前 {max_elements}/{len(arg)} 个键值对)")
        info.append(f"{indent_str}}}")
    
    # 其他类型（直接打印）
    else:
        info.append(f"{type(arg).__name__} (值: {str(arg)[:50]})")
    
    return '\n'.join(info)

def _get_return_var_name(func):
    """通过解析函数源码尝试获取返回值变量名（非100%准确）"""
    try:
        source = inspect.getsource(func)
        lines = [line.strip() for line in source.split('\n')]
        last_line = lines[-1]
        if 'return ' in last_line:
            return last_line.split('return ')[1].split(',')[0].strip()
    except:
        pass
    return ""
    
#######################################

# @my_trace
def random_choice(log_p):
    p = np.exp(log_p - np.max(log_p))  # softmax
    p /= np.sum(p)
    r = random.random() * sum(p)
    s = 0
    for i in range(len(p)):
        if r > s and r <= s + p[i]:
            return i, np.log(p[i])
        s += p[i]
    return len(p) - 1, np.log(p[len(p) - 1])


@attached
class Agent(BaseAgent):
    def __init__(self, agent_type="player", device=None, logger=None, monitor=None):
        super().__init__(agent_type, device, logger, monitor)

        self.model = NetworkModelActor()
        self.algorithm = Algorithm(device=device, logger=logger, monitor=monitor)
        self.preprocessor = Preprocessor()
        self.sample_manager = SampleManager()
        self.win_history = []
        self.logger = logger
        self.reset()

    def update_win_rate(self, is_win):
        self.win_history.append(is_win)
        if len(self.win_history) > 100:
            self.win_history.pop(0)
        return sum(self.win_history) / len(self.win_history) if len(self.win_history) > 10 else 0

    # @my_trace
    def _predict(self, obs, legal_action):
        with torch.no_grad():
            inputs = self.model.format_data(obs, legal_action) # send an array to model's format_data method, which turns array into tensor directly
            output_list = self.model(*inputs) # forward pass, which returns lots of useless info for "Actor" (actor and learner share the same network structure)

        np_output_list = []
        for output in output_list:
            np_output_list.append(output.numpy().flatten())

        return np_output_list

    # @my_trace
    def predict_process(self, feature, legal_action):
        feature = np.array([feature])
        legal_action = np.array([legal_action])
        log_probs, value = self._predict(feature, legal_action)
        return log_probs, value

    def observation_process(self, obs, extra_info=None):
        """
        基于当前帧观测信息+转移到当前帧的动作（上一动作），计算当前：
        1. 特征向量
        2. 合法动作
        3. 奖励
        并包装到 ObsData 中返回。
        
        注意：extra_info 不要用，因为 exploit 不能用它。只能提取 obs 里的信息。数据结构参考协议。
        """
        feature, legal_action, reward = self.preprocessor.process([obs, extra_info], self.last_action)

        return ObsData(
            feature=feature,
            legal_action=legal_action,
            reward=reward,
        )

    @predict_wrapper
    def predict(self, list_obs_data):
        feature = list_obs_data[0].feature
        legal_action = list_obs_data[0].legal_action
        log_probs, value = self.predict_process(feature, legal_action)
        
        action, log_prob = random_choice(log_probs)
        return [ActData(log_probs=log_probs, value=value, action=action, log_prob=log_prob)]

    def action_process(self, act_data):
        self.last_action = act_data.action # Update last action
        return act_data.action

    @exploit_wrapper
    def exploit(self, observation):
        obs_data = self.observation_process(observation["obs"], observation["extra_info"])
        feature = obs_data.feature
        legal_action = obs_data.legal_action
        log_probs, value = self.predict_process(feature, legal_action)
        action, log_prob = random_choice(log_probs)
        act = self.action_process(ActData(log_probs=log_probs, value=value, action=action, log_prob=log_prob))
        return act

    def reset(self):
        self.preprocessor.reset()
        self.last_prob = 0
        self.last_action = -1

    @learn_wrapper
    def learn(self, list_sample_data):
        self.algorithm.learn(list_sample_data)

    @save_model_wrapper
    def save_model(self, path=None, id="1"):
        # To save the model, it can consist of multiple files,
        # and it is important to ensure that each filename includes the "model.ckpt-id" field.
        # 保存模型, 可以是多个文件, 需要确保每个文件名里包括了model.ckpt-id字段
        model_file_path = f"{path}/model.ckpt-{str(id)}.pkl"
        torch.save(self.algorithm.model.state_dict(), model_file_path)
        self.logger.info(f"save model {model_file_path} successfully")

    @load_model_wrapper
    def load_model(self, path=None, id="1"):
        # When loading the model, you can load multiple files,
        # and it is important to ensure that each filename matches the one used during the save_model process.
        # 加载模型, 可以加载多个文件, 注意每个文件名需要和save_model时保持一致
        model_file_path = f"{path}/model.ckpt-{str(id)}.pkl"
        self.model.load_state_dict(torch.load(model_file_path, map_location="cpu"))
        self.logger.info(f"load model {model_file_path} successfully")
