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
from torch.optim.lr_scheduler import CosineAnnealingLR  # 导入调度器


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
                nn.init.constant_(param, 0)
    elif isinstance(layer, nn.Sequential):
        for sub_layer in layer:
            orthogonal_init(sub_layer, gain)


# --- DualLSTM 类 (修改后) ---
class DualLSTM(nn.Module):
    """
    双LSTM网络，分别处理状态特征和动作历史
    (修改后：状态使用LSTM，动作使用Embedding并直接拼接)
    """

    def __init__(self, state_input_dim, output_dim,
                 num_layers=1, hidden_dim=64, num_actions=16, action_embedding_dim=32, use_orthogonal_init=True):
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

        if use_orthogonal_init:
            print("------Applying orthogonal init to DualLSTM------")
            orthogonal_init(self.state_lstm)
            orthogonal_init(self.fusion_fc1)
            orthogonal_init(self.fusion_fc2)

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
        self.dual_lstm = DualLSTM(input_dim, output_dim, num_layers, hidden_dim,
                                  use_orthogonal_init=use_orthogonal_init)

        # 修改为多层感知机结构
        # Actor 输出层
        self.actor_output_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),  # 可以根据需要调整中间层维度
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        # Critic 输出层
        self.critic_output_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )

        # 应用正交初始化
        if use_orthogonal_init:
            print("------Applying orthogonal init to ActorCritic layers------")
            orthogonal_init(self.actor_output_layer[0], gain=1.0)
            orthogonal_init(self.actor_output_layer[2], gain=0.01)

            orthogonal_init(self.critic_output_layer[0], gain=1.0)
            orthogonal_init(self.critic_output_layer[2], gain=1.0)

    def forward(self, state_seq, action_seq):
        features = self.dual_lstm(state_seq, action_seq)
        action_probs = self.actor_output_layer(features)
        state_value = self.critic_output_layer(features)
        return action_probs, state_value

    def evaluate(self, state_seq, action_seq, action):
        action_probs, state_value = self.forward(state_seq, action_seq)
        dist = Categorical(logits=action_probs)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        return action_logprobs, state_value, dist_entropy

    def exploit(self, state_seq, action_seq, legal_actions):
        action_probs, _ = self.forward(state_seq, action_seq)
        mask = torch.ones_like(action_probs, dtype=torch.float32) * (-1e9)  # 初始化为负无穷大
        if legal_actions:
            mask[0, legal_actions] = 0.0  # 合法动作位置设置为0，不影响原始logits
        # action_probs 是 logits，直接在 logits 上进行掩码操作
        masked_logits = action_probs + mask
        action = torch.argmax(masked_logits, dim=-1)
        return action


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
            K_epochs=6,
            loss_weight={'actor': 0.6, 'critic': 0.6, 'entropy': 0.015},  # 0.015
            lstm_hidden_dim=64,
            lstm_num_layers=1,
            use_orthogonal_init=True,  # 新增参数
            total_training_steps=300  # 新增：用于调度器的总训练步数（或回合数）
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
        self.policy = ActorCritic(input_dim, output_dim, lstm_num_layers, lstm_hidden_dim,
                                  use_orthogonal_init=use_orthogonal_init)

        # 优化器参数分组需要调整以适应新的 MLP 结构
        # 遍历 Actor 和 Critic MLP 的子层以获取参数
        actor_params = [p for layer in self.policy.actor_output_layer for p in layer.parameters()]
        critic_params = [p for layer in self.policy.critic_output_layer for p in layer.parameters()]

        self.optimizer = torch.optim.Adam([
            {'params': actor_params, 'lr': lr_actor},
            {'params': critic_params, 'lr': lr_critic},
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
        action = self.policy.exploit(state_seq, self.action_seq, legal_actions)  # 修正：传递 action_seq
        self._action_process(action)
        return action.detach()

    def predict(self, state, done, legal_actions):
        state_seq = self._state_progress(state)
        # 在 Model.predict 中
        action_logits, state_value = self.policy(state_seq, self.action_seq)  # 修正：返回 logits

        # 创建一个负无穷大张量来屏蔽不合法动作
        illegal_action_mask = torch.ones_like(action_logits) * (-1e9)
        # 将合法动作位置设置为0，不影响原始logits
        if legal_actions:
            illegal_action_mask[0, legal_actions] = 0.0
        # 对 logits 应用掩码：将不合法动作的 logits 设置为一个非常小的负数 (趋近于负无穷)
        # 这样它们在 softmax 后对应的概率将趋近于 0
        masked_logits = action_logits + illegal_action_mask

        dist = Categorical(logits=masked_logits)  # 使用 logits
        action = dist.sample()

        # 使用原始（未屏蔽的）概率分布计算log_prob，以获得正确的梯度
        # 或者更准确地，使用masked_logits来计算log_prob，因为这也是你用于采样的方式
        log_prob = Categorical(logits=masked_logits).log_prob(action)  # 修正：使用 masked_logits

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

        returns_mean = np.mean(returns)
        returns_std = np.std(returns) + 1e-8
        returns = (returns - returns_mean) / returns_std

        return torch.tensor(advantages, dtype=torch.float32), torch.tensor(returns, dtype=torch.float32)

    # _calc_advantages 方法已不再需要，被 _compute_gae_and_returns 替代
    # def _calc_advantages(self, rewards, values): ...

    # _calc_loss 已更新，直接接收计算好的 advantages 和 returns
    def _calc_loss(self, episode, returns, advantages, logprobs, new_logprobs, new_values, entropy):
        """
        计算PPO损失。
        Args:
            returns (Tensor): GAE计算出的回报 (value function target)
            advantages (Tensor): GAE计算出的优势
            ... (其他参数不变)
        """
        # 计算策略比率
        clip_lowerbound = 0.001
        ratios = torch.exp(new_logprobs - logprobs.detach())

        # 归一化优势函数 (可选，但通常能稳定训练)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # 计算 Actor 损失 (Clipped Surrogate Objective)
        surr1 = ratios * advantages
        surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
        actor_loss = -torch.min(surr1, surr2).mean()
        #dual clip
        #actor_loss = -torch.max(min(surr1, surr2), clip_lowerbound * advantages).mean()

        # 计算 Critic 损失 (MSE)，目标是GAE计算出的 returns
        critic_loss = F.mse_loss(new_values.squeeze(), returns.detach())

        # 总损失
        loss = (self.loss_weight['actor'] * actor_loss
                + self.loss_weight['critic'] * critic_loss
                - self.loss_weight['entropy'] / (episode + 1) * entropy.mean())
        return loss

    # learn方法已重构，以集成GAE计算
    def learn(self, sample, last_state, last_done):

        if sample is None or not hasattr(sample, 'actions') or not sample.actions:
            # print("Skipping training cycle: no data received.")
            return

        rewards = np.array(sample.rewards, dtype=np.float32)
        # dones_np = np.array([b.item() for b in sample.dones], dtype=np.float32) # 如果dones是tensor列表
        dones_np = sample.dones.cpu().numpy() # 如果dones是单个tensor

        state_seqs = sample.state_seqs
        action_seqs = sample.action_seqs
        actions = sample.actions
        logprobs = sample.logprobs
        
        # 1. 计算 GAE 和 Returns
        with torch.no_grad():
            self.buffer['values'] = sample.values # GAE计算需要用到values
            # 注意：last_state 和 last_done 也应该从 sample 对象中获取
            last_state_from_sample = sample.last_state
            last_done_from_sample = sample.last_done

            last_state_seq = self._state_progress(last_state_from_sample)
            _, last_value_tensor = self.policy(last_state_seq, self.action_seq)
            last_value = last_value_tensor.cpu().item()

        advantages, returns = self._compute_gae_and_returns(rewards, dones_np, last_value)
        
        # 3. 多轮更新（PPO核心）
        data_size = len(actions)
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
                
                # 计算损失并反向传播
                loss = self._calc_loss(sample.episode, mb_returns, mb_advantages, mb_logprobs,
                                        new_logprobs, new_values, entropy)

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=2.0)
                self.optimizer.step()

        self.scheduler.step()
        self._clear_buffer()

    def save_model(self, path=None, id="1"):
        model_file_path = f"{path}/model.ckpt-{str(id)}.pt"
        torch.save({
            "policy": self.policy.state_dict(),
            "optimizer": self.optimizer.state_dict(),  # 保存优化器状态
            "scheduler": self.scheduler.state_dict()  # 保存调度器状态
        }, model_file_path)

    def load_model(self, path):
        checkpoint = torch.load(path)
        self.policy.load_state_dict(checkpoint["policy"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])  # 加载优化器状态
        self.scheduler.load_state_dict(checkpoint["scheduler"])  # 加载调度器状态
