#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2025 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""

import torch
from sympy.physics.units import action
from torch import nn
from torch.distributions import Categorical
import torch.nn.functional as F
import numpy as np
from kaiwu_agent.utils.common_func import attached


class DualLSTM(nn.Module):
    """
    双LSTM网络，分别处理状态特征和动作历史
    """
    def __init__(self, input_dim, output_dim, num_layers=1, hidden_dim=64):
        super(DualLSTM, self).__init__()
        
        # 状态LSTM - 处理环境状态
        self.state_lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        
        # 动作LSTM - 处理历史动作
        self.action_lstm = nn.LSTM(
            input_size=output_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        
        # 融合层 - 将两个LSTM的输出融合
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()
        )
        
        self.hidden_dim = hidden_dim
        
    def forward(self, state_seq, action_seq):
        # 初始化隐藏状态
        state_h0, state_c0 = self._init_hidden(state_seq.size(0))
        action_h0, action_c0 = self._init_hidden(action_seq.size(0))
        
        # 通过状态LSTM
        state_out, _ = self.state_lstm(state_seq, (state_h0, state_c0))
        state_out = state_out[:, -1, :]  # 取最后一个时间步的输出
        
        # 通过动作LSTM
        action_out, _ = self.action_lstm(action_seq, (action_h0, action_c0))
        action_out = action_out[:, -1, :]  # 取最后一个时间步的输出
        
        # 融合两个LSTM的输出
        combined = torch.cat([state_out, action_out], dim=1)
        fused_output = self.fusion(combined)
        
        return fused_output
    
    def _init_hidden(self, batch_size):
        # 初始化LSTM的隐藏状态
        return (torch.zeros(1, batch_size, self.hidden_dim),
                torch.zeros(1, batch_size, self.hidden_dim))

class ActorCritic(nn.Module):
    """
    基于Dual LSTM的Actor-Critic网络
    """
    def __init__(self, input_dim, output_dim, num_layers=1, hidden_dim=64):
        super(ActorCritic, self).__init__()
        
        self.dual_lstm = DualLSTM(input_dim, output_dim, num_layers, hidden_dim)
        
        # Actor网络 - 输出动作概率分布
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.Softmax()
        )
        
        # Critic网络 - 输出状态价值
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, state_seq, action_seq):
        # 通过Dual LSTM获取特征表示
        features = self.dual_lstm(state_seq, action_seq)
        
        # 获取动作概率分布
        action_probs = self.actor(features)
        
        # 获取状态价值
        state_value = self.critic(features)
        
        return action_probs, state_value
    
    def act(self, state_seq, action_seq, legal_actions):
        # 根据当前状态选择动作
        #print(legal_actions)
        action_probs, _ = self.forward(state_seq, action_seq)
        for i in range(len(action_probs[0])):
            if i not in legal_actions:
                action_probs[0][i]=0.000001
        current_sum = torch.sum(action_probs[0])

        if current_sum > 0:
            action_probs[0] = action_probs[0] / current_sum
        else:
            action_probs[0][0]=1#既然全部不合法了就随便取一个
        dist = Categorical(action_probs)
        action = dist.sample()
        action_logprob = dist.log_prob(action)
        #print("action",action)
        return action.detach(), action_logprob.detach()
    
    def evaluate(self, state_seq, action_seq, action):
        # 评估动作的价值和概率
        action_probs, state_value = self.forward(state_seq, action_seq)
        dist = Categorical(action_probs)
        
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        
        return action_logprobs, state_value, dist_entropy

    def exploit(self, state_seq, action_seq, legal_actions):
        # 选择最优动作
        action_probs, _ = self.forward(state_seq, action_seq)
        for i in range(len(action_probs[0])):
            if i not in legal_actions:
                action_probs[0][i] = 0.000001
        max_prob=0
        for j in range(len(action_probs[0])):
            if action_probs[0][j] > max_prob:
                action = torch.tensor([j])
                max_prob = action_probs[0][j]      
        ###############注意
        #dist = Categorical(action_probs)
        #action = dist.sample()
        ###############
        return action

class Model(nn.Module):
    def __init__(
        self, 
        input_dim, 
        output_dim,
        seq_length=64,
        lr_actor=1e-3,
        lr_critic=1e-3,
        lr_lstm=1e-3,
        eps_clip=0.2,
        K_epochs=10,
        loss_weight={'actor': 0.6, 'critic': 0.6, 'entropy': 0.2},
        lstm_hidden_dim=64,  # LSTM隐藏层维度
        lstm_num_layers=1    # LSTM层数
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.seq_length = seq_length
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.loss_weight = loss_weight
        self.lstm_hidden_dim = lstm_hidden_dim

        # ActorCritic网络
        self.policy = ActorCritic(input_dim, output_dim, lstm_num_layers, lstm_hidden_dim)
        # 优化器（分别为Actor和Critic设置学习率）
        self.optimizer = torch.optim.Adam([
            {'params': self.policy.actor.parameters(), 'lr': lr_actor},
            {'params': self.policy.critic.parameters(), 'lr': lr_critic},
            {'params': self.policy.dual_lstm.parameters(), 'lr': lr_lstm}
        ])

        self.state_seq = torch.zeros(1, seq_length, input_dim)
        self.action_seq = torch.zeros(1, seq_length, output_dim)
        self.buffer = {
            'state_seqs': [],          # 存储时序状态序列
            'action_seqs': [],
            'actions': [],
            'logprobs': [],
        }

    def _clear_buffer(self):
        """清空经验回放缓冲区"""
        self.buffer = {
            'state_seqs': [],          # 存储时序状态序列
            'action_seqs': [],
            'actions': [],
            'logprobs': [],
        }
        self.state_seq = torch.zeros(1, self.seq_length, self.input_dim)
        self.action_seq = torch.zeros(1, self.seq_length, self.output_dim)

    def reset(self):
        """重置模型的内部状态和缓冲区，为新的回合做准备"""
        self._clear_buffer()

    def _state_progress(self, state):
        """状态预处理（裁剪极端值和NaN）"""
        state = torch.FloatTensor(state)
        state = torch.clamp(state, min=-1e4, max=1e4)
        state = torch.nan_to_num(state, nan=0.0, posinf=1e4, neginf=-1e4)

        self.state_seq = torch.cat([self.state_seq[:, 1:, :], state.unsqueeze(0).unsqueeze(0)], dim=1)
        return self.state_seq

    def _action_process(self, action):
        one_hot_action = torch.zeros(1, self.output_dim)
        one_hot_action[0, action.item()] = 1
        self.action_seq = torch.cat([self.action_seq[:, 1:, :], one_hot_action.unsqueeze(0)], dim=1)

        return self.action_seq

    def exploit(self, state, legal_actions):
        """利用策略（选择概率最高的动作，用于测试）"""
        # 提取状态序列与动作序列
        state_seq = self._state_progress(state)
        # 获得最优动作
        action = self.policy.exploit(state_seq, self.action_seq, legal_actions)
        # 记录状态序列与动作序列
        action_seq = self._action_process(action)
        #dist = Categorical(action_probs)
        #action = dist.sample()
        #action=torch.tensor([2])
        return action.detach()

    def predict(self, state, legal_actions):
        """探索策略（采样动作，用于训练）"""
        # 提取状态序列与动作序列
        state_seq = self._state_progress(state)
        # 得到模型预测动作
        action, log_prob = self.policy.act(state_seq, self.action_seq, legal_actions)
        # 获得新动作序列
        action_seq = self._action_process(action)

        # 存储经验到缓冲区（包括隐藏状态，用于训练）
        self.buffer['state_seqs'].append(state_seq.squeeze(0))
        self.buffer['action_seqs'].append(action_seq.squeeze(0))
        self.buffer['actions'].append(action.detach())
        self.buffer['logprobs'].append(log_prob.detach())
        
        return action.detach()

    def _calc_advantages(self, rewards, values):
        """计算优势函数"""
        #print("reward:",rewards)
        #print("value:",values)
        return rewards.detach() - values.detach()

    def _calc_loss(self, rewards, logprobs, new_logprobs, new_values, entropy):
        """计算PPO损失（适配LSTM输出）"""
        # 计算优势
        advantages = self._calc_advantages(rewards, new_values)
        # 计算策略比率
        ratios = torch.exp(new_logprobs - logprobs)
        # 计算损失
        surr1 = ratios * advantages
        surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
        
        actor_loss = torch.mean(-torch.min(surr1, surr2))
        critic_loss = torch.mean(F.mse_loss(new_values.squeeze(), rewards))  # 确保维度匹配
        
        # 总损失： Actor损失 + Critic损失 - 熵奖励（鼓励探索）
        loss = (self.loss_weight['actor'] * actor_loss 
                + self.loss_weight['critic'] * critic_loss 
                - self.loss_weight['entropy'] * entropy.mean())
        return loss

    def learn(self, sample):
        """训练PPO（适配LSTM的批量时序数据），引入minibatch"""
        # 1. 准备训练数据
        rewards = torch.stack([torch.tensor(reward, dtype=torch.float32) for reward in sample.rewards])
        state_seqs = torch.stack(self.buffer['state_seqs'])
        action_seqs = torch.stack(self.buffer['action_seqs'])
        actions = torch.stack(self.buffer['actions'])
        logprobs = torch.stack(self.buffer['logprobs'])

        # 2. 处理奖励归一化
        if rewards.std() > 1e-7:
            rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        else:
            rewards = rewards - rewards.mean()

        # 将所有数据打包，方便后续打乱和分割
        # 注意：这里假设所有张量的第一个维度都是批次大小，并且它们长度相同
        data_size = state_seqs.size(0)
        # 定义minibatch大小，可以作为hyperparameter
        minibatch_size = 32  # 表示划分的大小

        # 3. 多轮更新（PPO核心）
        for i in range(self.K_epochs):
            # 在每个epoch开始时，打乱数据索引
            indices = np.arange(data_size)
            np.random.shuffle(indices)

            # 遍历minibatch
            for start_idx in range(0, data_size, minibatch_size):
                end_idx = min(start_idx + minibatch_size, data_size)
                batch_indices = indices[start_idx:end_idx]

                # 获取当前minibatch的数据
                mb_state_seqs = state_seqs[batch_indices]
                mb_action_seqs = action_seqs[batch_indices]
                mb_actions = actions[batch_indices]
                mb_logprobs = logprobs[batch_indices]
                mb_rewards = rewards[batch_indices]

                # 评估旧动作和状态
                new_logprobs, new_values, entropy = self.policy.evaluate(
                    mb_state_seqs, mb_action_seqs, mb_actions
                )

                # 计算损失并反向传播
                loss = self._calc_loss(mb_rewards.squeeze(), mb_logprobs.squeeze(),
                                       new_logprobs, new_values, entropy)

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=2.0)  # 全局梯度裁剪
                self.optimizer.step()

                # 清空缓冲区，准备下一轮收集经验
        self._clear_buffer()

    def save_model(self, path=None, id="1",save_model=True):
        model_file_path = f"{path}/model.ckpt-{str(id)}.pt"
        torch.save({
            "policy": self.policy.state_dict()
        }, model_file_path)

    def load_model(self, path,load_model=True):
        checkpoint = torch.load(path)
        self.policy.load_state_dict(checkpoint["policy"])
