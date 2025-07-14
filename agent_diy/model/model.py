#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2025 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""

import torch
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
        self.num_layers = num_layers

        # 增加隐藏状态缓存
        self.state_hidden = None
        self.action_hidden = None
    def reset_hidden(self, batch_size=1):
        self.state_hidden = self._init_hidden(batch_size)
        self.action_hidden = self._init_hidden(batch_size)    

    def forward(self, state_seq, action_seq):
        # 使用正确的隐藏状态元组
        state_out, (state_hn, state_cn) = self.state_lstm(state_seq, self.state_hidden)
        action_out, (action_hn, action_cn) = self.action_lstm(action_seq, self.action_hidden)
        
        # 更新缓存
        self.state_hidden = (state_hn.detach(), state_cn.detach())
        self.action_hidden = (action_hn.detach(), action_cn.detach())
        
        # 直接使用第一次计算的输出
        state_out = state_out[:, -1, :]
        action_out = action_out[:, -1, :]
        
        combined = torch.cat([state_out, action_out], dim=1)
        return self.fusion(combined)
        
    def _init_hidden(self, batch_size):
        # 初始化LSTM的隐藏状态
        return (torch.zeros(self.num_layers, batch_size, self.hidden_dim),
                torch.zeros(self.num_layers, batch_size, self.hidden_dim))

class ActorCritic(nn.Module):
    """
    基于Dual LSTM的Actor-Critic网络
    """
    def __init__(self, input_dim, output_dim, num_layers=1, hidden_dim=64):
        super(ActorCritic, self).__init__()
        
        self.dual_lstm = DualLSTM(input_dim, output_dim, num_layers, hidden_dim)
        
        # Actor网络 - 输出动作概率分布
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Softmax()
        )
        
        # Critic网络 - 输出状态价值
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
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
        action_probs, _ = self.forward(state_seq, action_seq)
        filtered_action_probs = self._filter_legal_actions(action_probs, legal_actions)
        # print(filtered_action_probs)
        dist = Categorical(filtered_action_probs)
        action = dist.sample()
        action_logprob = dist.log_prob(action)
        
        return action.detach(), action_logprob.detach()

    def evaluate(self, state_seq, next_state_seq, action_seq, next_action_seq, action):
        # 当前状态使用当前动作序列
        action_probs, state_value = self.forward(state_seq, action_seq)
        # 下一个状态使用更新后的动作序列
        _, next_state_value = self.forward(next_state_seq, next_action_seq)
        
        dist = Categorical(action_probs)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        
        return action_logprobs, state_value, next_state_value, dist_entropy

    def exploit(self, state_seq, action_seq, legal_actions):
        # 选择最优动作
        action_probs, _ = self.forward(state_seq, action_seq)
        filtered_action_probs = self._filter_legal_actions(action_probs, legal_actions)
        action = torch.argmax(filtered_action_probs, dim=-1)
        return action

    def _filter_legal_actions(self, action_probs, legal_actions):
        mask = torch.zeros_like(action_probs)
        mask[0, legal_actions] = 1.0
        masked_probs = action_probs * mask
        # 处理全零情况
        if masked_probs.sum() < 1e-6:
            masked_probs[legal_actions] = 1.0 / len(legal_actions)
        return masked_probs / masked_probs.sum()


class Model(nn.Module):
    def __init__(
        self, 
        input_dim, 
        output_dim,
        minibatch=180,
        gamma=0.95,
        lam=0.9,
        seq_length=64,
        lr_actor=3e-4,
        lr_critic=3e-4,
        lr_lstm=3e-4,
        eps_clip=0.2,
        K_epochs=10,
        loss_weight={'actor': 1.0, 'critic': 0.5, 'entropy': 0.01},
        lstm_hidden_dim=64,  # LSTM隐藏层维度
        lstm_num_layers=1    # LSTM层数
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.minibatch = minibatch
        self.gamma = gamma
        self.lam = lam
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

    def reset(self, batch_size=1):
        """重置LSTM状态和序列缓存"""
        self.state_seq = torch.zeros(1, self.seq_length, self.input_dim)
        self.action_seq = torch.zeros(1, self.seq_length, self.output_dim)
        self.buffer = {k: [] for k in self.buffer}  # 清空缓冲区
        self.policy.dual_lstm.reset_hidden(batch_size)  # 重置LSTM隐藏状态

    def _state_progress(self, state):
        """状态预处理（裁剪极端值和NaN）"""
        state = torch.FloatTensor(state)
        state = torch.clamp(state, min=-1e4, max=1e4)
        state = torch.nan_to_num(state, nan=0.0, posinf=1e4, neginf=-1e4)
        state = (state - state.mean()) / (state.std() + 1e-8)

        self.state_seq = torch.cat([self.state_seq[:, 1:, :], state.unsqueeze(0).unsqueeze(0)], dim=1)
        return self.state_seq

    def _action_process(self, action):
        # 创建(1, output_dim)的one-hot向量
        one_hot_action = F.one_hot(action, num_classes=self.output_dim).float().view(1, 1, -1)
        # (1, seq_length, output_dim)
        self.action_seq = torch.cat([self.action_seq[:, 1:], one_hot_action], dim=1)
        return self.action_seq

    def exploit(self, state, legal_actions):
        """利用策略（选择概率最高的动作，用于测试）"""
        # 提取状态序列与动作序列
        state_seq = self._state_progress(state)
        # 获得最优动作
        action = self.policy.exploit(state_seq, self.action_seq, legal_actions)
        # 记录状态序列与动作序列
        action_seq = self._action_process(action)

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

    def _calc_advantages(self, values, next_values, rewards, dones):
        advantages = torch.zeros_like(rewards)
        returns = torch.zeros_like(rewards)
        gae = 0
        
        # 逆序计算GAE
        for t in reversed(range(rewards.size(0))):
            delta = rewards[t] + self.gamma * (1 - dones[t]) * next_values[t] - values[t]
            gae = delta + self.gamma * self.lam * (1 - dones[t]) * gae
            advantages[t] = gae
        
        # 计算回报
        returns = advantages + values
        
        # 归一化
        advantages = self._normalize(advantages)
        return advantages, returns

    def _calc_loss(self, rewards, logprobs, new_logprobs, new_values, new_next_values, entropy, dones):
        """计算PPO损失（适配LSTM输出）"""
        # 计算优势
        advantages, returns = self._calc_advantages(new_values.squeeze(-1), new_next_values.squeeze(-1), rewards, dones)
        # 计算策略比率
        ratios = torch.exp(new_logprobs - logprobs)
        # 计算损失
        surr1 = ratios * advantages
        surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
        
        actor_loss = torch.mean(-torch.min(surr1, surr2))
        critic_loss = torch.mean(F.mse_loss(returns, new_values.squeeze(-1)))
        
        # 总损失： Actor损失 + Critic损失 - 熵奖励（鼓励探索）
        loss = (self.loss_weight['actor'] * actor_loss 
                + self.loss_weight['critic'] * critic_loss 
                - self.loss_weight['entropy'] * entropy.mean())
        return loss

    def _normalize(self, data):
        return (data - data.mean()) / (data.std() + 1e8)

    def learn(self, sample):
        """训练PPO（适配LSTM的批量时序数据）"""
        # 准备训练数据
        rewards =torch.stack([torch.tensor(reward, dtype=torch.float32) for reward in sample.rewards])
        dones = torch.stack([torch.tensor(done, dtype=torch.float32) for done in sample.dones])
        state_seqs = torch.stack(self.buffer['state_seqs'][:-1])
        next_state_seqs = torch.stack(self.buffer['state_seqs'][1:])
        action_seqs = torch.stack(self.buffer['action_seqs'][:-1])
        next_action_seqs = torch.stack(self.buffer['action_seqs'][1:])
        actions = torch.stack(self.buffer['actions'][:-1])
        logprobs = torch.stack(self.buffer['logprobs'][:-1])
        
        data_size = state_seqs.size(0)
        # 多轮更新（PPO核心）
        for i in range(self.K_epochs):
            indices = np.arange(data_size)
            np.random.shuffle(indices)
            # 遍历minibatch
            for start_idx in range(0, data_size, self.minibatch):
                end_idx = min(start_idx + self.minibatch, data_size)
                batch_indices = indices[start_idx:end_idx]
                # 重置网络结构
                self.reset(end_idx - start_idx)
                # 获取当前minibatch的数据
                mb_state_seqs = state_seqs[batch_indices]
                mb_next_state_seqs = next_state_seqs[batch_indices]
                mb_action_seqs = action_seqs[batch_indices]
                mb_next_action_seqs = next_action_seqs[batch_indices]
                mb_actions = actions[batch_indices]
                mb_logprobs = logprobs[batch_indices]
                mb_rewards = self._normalize(rewards[batch_indices])
                mb_dones = dones[batch_indices]
                
                # 评估动作和状态
                new_logprobs, new_values, new_next_values, entropy = self.policy.evaluate(
                    mb_state_seqs, mb_next_state_seqs, mb_action_seqs, mb_next_action_seqs, mb_actions
                )
                # 计算损失并反向传播
                loss = self._calc_loss(mb_rewards, mb_logprobs,
                                       new_logprobs, new_values, new_next_values, entropy, mb_dones)

                self.optimizer.zero_grad()
                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    self.parameters(), 
                    max_norm=5.0,
                    norm_type=2.0  # L2范数裁剪
                )
                self.optimizer.step()
        
    def save_model(self, path=None, id="1"):
        model_file_path = f"{path}/model.ckpt-{str(id)}.pt"
        torch.save({
            "policy": self.policy.state_dict()
        }, model_file_path)

    def load_model(self, path=None, id="1"):
        model_file_path = f"{path}/model.ckpt-{str(id)}.pt"
        checkpoint = torch.load(model_file_path)
        self.policy.load_state_dict(checkpoint["policy"])
        self._clear_buffer()