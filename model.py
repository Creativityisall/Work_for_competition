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
from torch.optim.lr_scheduler import CosineAnnealingLR # 导入调度器

def orthogonal_init(layer, gain=1.0):
    if isinstance(layer, nn.Linear):
        nn.init.orthogonal_(layer.weight, gain=gain)
        if layer.bias is not None:
            nn.init.constant_(layer.bias, 0)
    elif isinstance(layer, nn.LSTM):
        # LSTM 有多个权重和偏置：
        # ih_l[k]：第 k 层输入到隐藏状态的权重
        # hh_l[k]：第 k 层隐藏状态到隐藏状态的权重
        # (以及对应的偏置 ih_b 和 hh_b)
        for name, param in layer.named_parameters():
            if 'weight' in name:
                nn.init.orthogonal_(param, gain=gain)
            elif 'bias' in name:
                nn.init.constant_(param, 0) # LSTM偏置通常初始化为0，或根据门控机制初始化为小正值

# --- DualLSTM 类 (修改后) ---
class DualLSTM(nn.Module):
    """
    双LSTM网络，分别处理状态特征和动作历史
    (修改后：状态使用LSTM，动作使用Embedding并直接拼接)
    """

    def __init__(self, state_input_dim, output_dim,
                 num_layers=1, hidden_dim=64, num_actions=4, action_embedding_dim=8, use_orthogonal_init=True):
        super(DualLSTM, self).__init__()
        self.state_lstm = nn.LSTM(
            input_size=state_input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        self.action_embedding = nn.Embedding(num_actions, action_embedding_dim)
        self.fusion_fc1 = nn.Linear(hidden_dim + action_embedding_dim, hidden_dim)
        self.fusion_fc2 = nn.Linear(hidden_dim, hidden_dim)

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # 应用正交初始化
        if use_orthogonal_init:
            print("------Applying orthogonal init to DualLSTM------")
            orthogonal_init(self.state_lstm)
            orthogonal_init(self.fusion_fc1)
            orthogonal_init(self.fusion_fc2)
            # Embedding层通常不需要正交初始化，它们的权重通常随机初始化，并在训练中学习

    def _init_hidden(self, batch_size):
        return (torch.zeros(self.num_layers, batch_size, self.hidden_dim),
                torch.zeros(self.num_layers, batch_size, self.hidden_dim))

    def forward(self, state_seq, action_seq):
        state_h0, state_c0 = self._init_hidden(state_seq.size(0))
        state_out, _ = self.state_lstm(state_seq, (state_h0, state_c0))
        state_out = state_out[:, -1, :]
        last_action_id = action_seq[:, -1, :]
        last_action_id = torch.argmax(last_action_id, dim=-1)
        action_embedded = self.action_embedding(last_action_id.long())
        combined = torch.cat([state_out, action_embedded], dim=1)

        fused_output = torch.tanh(self.fusion_fc1(combined))
        fused_output = torch.tanh(self.fusion_fc2(fused_output))
        return fused_output


# --- ActorCritic 类 (修改后) ---
class ActorCritic(nn.Module):
    """
    基于Dual LSTM的Actor-Critic网络
    """

    def __init__(self, input_dim, output_dim, num_layers=1, hidden_dim=64, use_orthogonal_init=True):
        super(ActorCritic, self).__init__()
        self.dual_lstm = DualLSTM(input_dim, output_dim, num_layers, hidden_dim, use_orthogonal_init=use_orthogonal_init)

        self.actor_output_layer = nn.Linear(hidden_dim, output_dim)
        self.critic_output_layer = nn.Linear(hidden_dim, 1)

        # 应用正交初始化
        if use_orthogonal_init:
            #print("------Applying orthogonal init to ActorCritic layers------")
            # Actor 输出层使用 gain=0.01
            orthogonal_init(self.actor_output_layer, gain=0.01)
            # Critic 输出层使用默认 gain=1.0
            orthogonal_init(self.critic_output_layer)

    def forward(self, state_seq, action_seq):
        features = self.dual_lstm(state_seq, action_seq)
        action_probs = torch.softmax(self.actor_output_layer(features), dim=-1) # Softmax应用于输出层
        state_value = self.critic_output_layer(features)
        return action_probs, state_value

    def evaluate(self, state_seq, action_seq, action):
        action_probs, state_value = self.forward(state_seq, action_seq)
        dist = Categorical(action_probs)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        return action_logprobs, state_value, dist_entropy

    def exploit(self, state_seq, action_seq, legal_actions):
        # 选择最优动作
        action_probs, _ = self.forward(state_seq, action_seq)
        action_probs = action_probs.detach()

        # 根据合法动作进行过滤和重新归一化
        mask = torch.zeros_like(action_probs)
        for i in legal_actions:
            mask[0][i] = 1.0 # 确保合法动作的mask为1

        action_probs_masked = action_probs * mask

        current_sum = torch.sum(action_probs_masked)
        if current_sum > 0:
            action_probs_masked = action_probs_masked / current_sum
        else:
            # 如果所有合法动作概率都是0（例如，被masking后），则均匀分配给合法动作
            # 避免除以零和不确定的行为
            if len(legal_actions) > 0:
                for i in legal_actions:
                    action_probs_masked[0][i] = 1.0 / len(legal_actions)
            else:
                # 如果没有合法动作，则可能采取一个默认动作或报错
                # 这里假设至少有一个合法动作，或者模型会处理这种情况
                pass # 根据实际需求处理无合法动作的情况

        dist = Categorical(action_probs_masked)
        action = dist.sample()
        return action.detach()


# --- Model 类 (添加调度器) ---
class Model(nn.Module):
    def __init__(
            self,
            input_dim,
            output_dim,
            seq_length=64,
            lr_actor=1e-4,
            lr_critic=1e-4,
            lr_lstm=1e-4,
            gamma=0.99,  # GAE新增：折扣因子
            gae_lambda=0.95,  # GAE新增：GAE的lambda参数
            eps_clip=0.2,
            K_epochs=4,
            loss_weight={'actor': 0.6, 'critic': 0.6, 'entropy': 1},#0.015
            lstm_hidden_dim=64,
            lstm_num_layers=1,
            use_orthogonal_init=True, # 新增参数
            total_training_steps=300 # 新增：用于调度器的总训练步数（或回合数）
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.seq_length = seq_length
        self.gamma = gamma  # GAE新增
        self.gae_lambda = gae_lambda  # GAE新增
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.loss_weight = loss_weight
        self.lstm_hidden_dim = lstm_hidden_dim

        # 将 use_orthogonal_init 传递给 ActorCritic
        self.policy = ActorCritic(input_dim, output_dim, lstm_num_layers, lstm_hidden_dim, use_orthogonal_init=use_orthogonal_init)
        self.optimizer = torch.optim.Adam([
            {'params': self.policy.actor_output_layer.parameters(), 'lr': lr_actor}, # 更改为 actor_output_layer
            {'params': self.policy.critic_output_layer.parameters(), 'lr': lr_critic}, # 更改为 critic_output_layer
            {'params': self.policy.dual_lstm.parameters(), 'lr': lr_lstm}
        ])

        # 初始化学习率调度器
        # 我们使用 CosineAnnealingLR，T_max 设置为总训练步数/回合数
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=total_training_steps)


        self.state_seq = torch.zeros(1, seq_length, input_dim)
        self.action_seq = torch.zeros(1, seq_length, output_dim)
        self.seq_idx = 0

        # 初始化缓冲区，为GAE添加 'values' 和 'dones'
        self._clear_buffer()

    def _clear_buffer(self):
        """清空经验回放缓冲区"""
        self.buffer = {
            'state_seqs': [],
            'action_seqs': [],
            'actions': [],
            'logprobs': [],
            'values': [],  # GAE新增：存储每个状态的价值
            'dones': []  # GAE新增：存储每个时间步的完成标志
        }
        self.state_seq.zero_()
        self.action_seq.zero_()
        self.seq_idx = 0

    def reset(self):
        self._clear_buffer()

    def _state_progress(self, state):
        state = torch.FloatTensor(state)
        state = torch.clamp(state, min=-1e4, max=1e4)
        state = torch.nan_to_num(state, nan=0.0, posinf=1e4, neginf=-1e4)
        if self.seq_idx < self.seq_length:
            self.state_seq[0, self.seq_idx, :] = state
        else:
            self.state_seq[0, :-1, :] = self.state_seq[0, 1:, :].clone()
            self.state_seq[0, -1, :] = state

        if self.seq_idx < self.seq_length:
            self.seq_idx += 1

        return self.state_seq

    def _action_process(self, action):
        one_hot_action = torch.zeros(1, self.output_dim)
        one_hot_action[0, action.item()] = 1
        current_idx = min(self.seq_idx - 1, self.seq_length - 1)
        self.action_seq[0, current_idx, :] = one_hot_action
        return self.action_seq

    def exploit(self, state, legal_actions):
        state_seq = self._state_progress(state)
        action = self.policy.exploit(state_seq, self.action_seq, legal_actions)
        self._action_process(action)
        return action.detach()

    def predict(self, state, done, legal_actions):
        """探索策略（采样动作，用于训练），并存储GAE所需信息"""
        state_seq = self._state_progress(state)

        action_probs, state_value = self.policy(state_seq, self.action_seq)

        action_probs_detached = action_probs.detach()
        mask = torch.zeros_like(action_probs_detached)
        for i in legal_actions:
            mask[0][i] = 1.0

        action_probs_masked = action_probs_detached * mask

        # 重新归一化概率
        current_sum = torch.sum(action_probs_masked)
        if current_sum > 0:
            action_probs_masked /= current_sum
        else:
            # 如果所有合法动作概率都为0，均匀分配给合法动作
            if len(legal_actions) > 0:
                for i in legal_actions:
                    action_probs_masked[0][i] = 1.0 / len(legal_actions)
            else:
                pass # 同 exploit 方法中的处理

        dist = Categorical(action_probs_masked)
        action = dist.sample()
        log_prob = Categorical(action_probs).log_prob(action) # 从原始分布计算log_prob

        action_seq = self._action_process(action)

        self.buffer['state_seqs'].append(state_seq.squeeze(0))
        self.buffer['action_seqs'].append(action_seq.squeeze(0))
        self.buffer['actions'].append(action.detach())
        self.buffer['logprobs'].append(log_prob.detach())
        self.buffer['values'].append(state_value.detach())
        self.buffer['dones'].append(torch.tensor(done, dtype=torch.float32))

        return action.detach()

    def _compute_gae_and_returns(self, rewards, dones, last_value):
        """
        根据收集到的经验，计算每个时间步的优势函数（GAE）和回报（Returns）。
        """
        values = torch.stack(self.buffer['values']).squeeze().cpu().numpy()
        buffer_size = len(self.buffer['actions'])

        advantages = np.zeros(buffer_size, dtype=np.float32)
        last_gae_lam = 0

        for step in reversed(range(buffer_size)):
            if step == buffer_size - 1:
                next_non_terminal = 1.0 - dones[-1]
                next_values = last_value
            else:
                next_non_terminal = 1.0 - self.buffer['dones'][step + 1].cpu().numpy()
                next_values = values[step + 1]

            delta = rewards[step] + self.gamma * next_values * next_non_terminal - values[step]
            last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
            advantages[step] = last_gae_lam

        returns = advantages + values

        adv_mean = np.mean(advantages)
        adv_std = np.std(advantages) + 1e-8
        advantages = (advantages - adv_mean) / adv_std

        #returns_mean = np.mean(returns)
        #returns_std = np.std(returns) + 1e-8
        #returns = (returns - returns_mean) / returns_std

        return torch.tensor(advantages, dtype=torch.float32), torch.tensor(returns, dtype=torch.float32)

    # _calc_advantages 方法已不再需要，被 _compute_gae_and_returns 替代
    # def _calc_advantages(self, rewards, values): ...

    # _calc_loss 已更新，直接接收计算好的 advantages 和 returns
    def _calc_loss(self,episode, returns, advantages, logprobs, new_logprobs, new_values, entropy):
        """
        计算PPO损失。
        Args:
            returns (Tensor): GAE计算出的回报 (value function target)
            advantages (Tensor): GAE计算出的优势
            ... (其他参数不变)
        """
        # 计算策略比率
        ratios = torch.exp(new_logprobs - logprobs)

        # 归一化优势函数 (可选，但通常能稳定训练)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # 计算 Actor 损失 (Clipped Surrogate Objective)
        surr1 = ratios * advantages
        surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
        actor_loss = -torch.min(surr1, surr2).mean()

        # 计算 Critic 损失 (MSE)，目标是GAE计算出的 returns
        critic_loss = F.mse_loss(new_values.squeeze(), returns)

        # 总损失
        loss = (self.loss_weight['actor'] * actor_loss
                + self.loss_weight['critic'] * critic_loss
                - self.loss_weight['entropy']/ (episode + 1) * entropy.mean())
        return loss

    # learn方法已重构，以集成GAE计算
    def learn(self, sample, last_state, last_done):
        """
        训练PPO模型。
        Args:
            sample: 从外部传入的样本数据，我们主要使用 sample.rewards。
            last_state: 序列收集完成后，环境返回的最后一个状态。用于计算bootstrap value。
            last_done: 最后一个状态是否是终止状态。
        """
        # 1. 准备训练数据
        rewards = np.array(sample.rewards, dtype=np.float32)
        dones = np.array([b.item() for b in self.buffer['dones']], dtype=np.float32)

        state_seqs = torch.stack(self.buffer['state_seqs'])
        action_seqs = torch.stack(self.buffer['action_seqs'])
        actions = torch.stack(self.buffer['actions']).squeeze()
        logprobs = torch.stack(self.buffer['logprobs']).squeeze()

        # 2. 计算 GAE 和 Returns
        # 首先，需要计算最后一个状态的价值(bootstrap value)，用于GAE计算
        with torch.no_grad():
            # 需要为 last_state 构建它对应的 state_seq 和 action_seq
            last_state_seq = self._state_progress(last_state)  # 这会更新 self.state_seq
            # 注意：last_action_seq 理论上应该是导致 last_state 的那个动作的序列，
            # self.action_seq 此时已经包含了最后一个真实动作，所以可以直接使用。
            _, last_value_tensor = self.policy(last_state_seq, self.action_seq)
            last_value = last_value_tensor.cpu().item()

        advantages, returns = self._compute_gae_and_returns(rewards, dones, last_value)

        # 将数据移动到设备
        # device = next(self.parameters()).device
        # advantages = advantages.to(device)
        # returns = returns.to(device)

        # 3. 多轮更新（PPO核心）
        data_size = len(self.buffer['actions'])
        minibatch_size = 512

        for _ in range(self.K_epochs):
            indices = np.arange(data_size)
            np.random.shuffle(indices)

            for start_idx in range(0, data_size, minibatch_size):
                end_idx = min(start_idx + minibatch_size, data_size)
                batch_indices = indices[start_idx:end_idx]

                # 获取当前minibatch的数据
                mb_state_seqs = state_seqs[batch_indices]
                mb_action_seqs = action_seqs[batch_indices]
                mb_actions = actions[batch_indices]
                mb_logprobs = logprobs[batch_indices]
                mb_advantages = advantages[batch_indices]
                mb_returns = returns[batch_indices]

                # 评估旧动作和状态
                new_logprobs, new_values, entropy = self.policy.evaluate(
                    mb_state_seqs, mb_action_seqs, mb_actions
                )
                episode = sample.episode
                # 计算损失并反向传播
                loss = self._calc_loss(episode, mb_returns, mb_advantages, mb_logprobs,
                                         new_logprobs, new_values, entropy)

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=2.0)
                self.optimizer.step()

        # 在每次学习迭代后，更新学习率调度器
        self.scheduler.step()

        # 清空缓冲区，准备下一轮收集经验
        self._clear_buffer()

    def save_model(self, path=None, id="1"):
        model_file_path = f"{path}/model.ckpt-{str(id)}.pt"
        torch.save({
            "policy": self.policy.state_dict(),
            "optimizer": self.optimizer.state_dict(), # 保存优化器状态
            "scheduler": self.scheduler.state_dict() # 保存调度器状态
        }, model_file_path)

    def load_model(self, path):
        checkpoint = torch.load(path)
        self.policy.load_state_dict(checkpoint["policy"])
        self.optimizer.load_state_dict(checkpoint["optimizer"]) # 加载优化器状态
        self.scheduler.load_state_dict(checkpoint["scheduler"]) # 加载调度器状态
