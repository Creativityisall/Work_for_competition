# agent_diy/model/model.py
#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
############################################################################
# Copyright © 1998 - 2025 Tencent.
# All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""
import torch
from torch import nn
from torch.distributions import Categorical
import torch.nn.functional as F
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingLR

def orthogonal_init(layer, gain=1.0):
    if isinstance(layer, nn.Linear):
        nn.init.orthogonal_(layer.weight, gain=gain)
        if layer.bias is not None:
            nn.init.constant_(layer.bias, 0)
    elif isinstance(layer, nn.LSTM):
        for name, param in layer.named_parameters():
            if 'weight' in name:
                nn.init.orthogonal_(param, gain=gain)
            elif 'bias' in name:
                nn.init.constant_(param, 0)
    elif isinstance(layer, nn.Sequential):
        for sub_layer in layer:
            orthogonal_init(sub_layer, gain)

# --- DualLSTM 类 (保持不变) ---
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

# --- ActorCritic 类 (保持不变) ---
class ActorCritic(nn.Module):
    """
    基于Dual LSTM的Actor-Critic网络
    """
    def __init__(self, input_dim, output_dim, num_layers=1, hidden_dim=64, use_orthogonal_init=True):
        super(ActorCritic, self).__init__()
        self.dual_lstm = DualLSTM(input_dim, output_dim, num_layers, hidden_dim,
                                  use_orthogonal_init=use_orthogonal_init)
        self.actor_output_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        self.critic_output_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )

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
        mask = torch.ones_like(action_probs, dtype=torch.float32) * (-1e9)
        if legal_actions:
            mask[0, legal_actions] = 0.0
        masked_logits = action_probs + mask
        action = torch.argmax(masked_logits, dim=-1)
        return action

# --- Model 类 (修改后，移除buffer和GAE计算) ---
class Model(nn.Module):
    def __init__(
            self,
            input_dim,
            output_dim,
            seq_length=64,
            lr_actor=1e-4,
            lr_critic=1e-4,
            lr_lstm=1e-4,
            eps_clip=0.2,
            K_epochs=6,
            loss_weight={'actor': 0.6, 'critic': 0.6, 'entropy': 0.015},
            lstm_hidden_dim=64,
            lstm_num_layers=1,
            use_orthogonal_init=True,
            total_training_steps=300
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.seq_length = seq_length
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.loss_weight = loss_weight
        self.lstm_hidden_dim = lstm_hidden_dim

        self.policy = ActorCritic(input_dim, output_dim, lstm_num_layers, lstm_hidden_dim,
                                    use_orthogonal_init=use_orthogonal_init)

        actor_params = [p for layer in self.policy.actor_output_layer for p in layer.parameters()]
        critic_params = [p for layer in self.policy.critic_output_layer for p in layer.parameters()]

        self.optimizer = torch.optim.Adam([
            {'params': actor_params, 'lr': lr_actor},
            {'params': critic_params, 'lr': lr_critic},
            {'params': self.policy.dual_lstm.parameters(), 'lr': lr_lstm}
        ])

        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=total_training_steps)

        # 状态追踪序列，但不再是数据收集的buffer
        self.state_seq = torch.zeros(1, seq_length, input_dim)
        self.action_seq = torch.zeros(1, seq_length, output_dim)
        self.seq_idx = 0

    def reset(self):
        """仅重置序列追踪的状态，不涉及数据buffer"""
        self.state_seq.zero_()
        self.action_seq.zero_()
        self.seq_idx = 0

    def _state_progress(self, state):
        state = torch.FloatTensor(state)
        state = torch.clamp(state, min=-1e4, max=1e4)
        state = torch.nan_to_num(state, nan=0.0, posinf=1e4, neginf=-1e4)
        if self.seq_idx < self.seq_length:
            self.state_seq[0, self.seq_idx, :] = state
        else:
            # self.state_seq[0, :-1, :] = self.state_seq[0, 1:, :].clone() # 原始行
            self.state_seq[0, :self.seq_length-1, :] = self.state_seq[0, 1:, :].clone() # 修复索引以避免越界
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
        """
        执行预测，返回动作、log_prob和价值。不再收集数据。
        """
        state_seq = self._state_progress(state)
        action_logits, state_value = self.policy(state_seq, self.action_seq)

        illegal_action_mask = torch.ones_like(action_logits) * (-1e9)
        if legal_actions:
            illegal_action_mask[0, legal_actions] = 0.0
        
        masked_logits = action_logits + illegal_action_mask
        dist = Categorical(logits=masked_logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        action_seq = self._action_process(action)
        
        # 返回所有需要被外部收集器记录的数据
        return (action.detach(), log_prob.detach(), state_value.detach(), 
                state_seq.squeeze(0), action_seq.squeeze(0))

    def _calc_loss(self, episode, returns, advantages, logprobs, new_logprobs, new_values, entropy):
        """计算PPO损失 (此函数保持不变)"""
        ratios = torch.exp(new_logprobs - logprobs.detach())
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        surr1 = ratios * advantages
        surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
        actor_loss = -torch.min(surr1, surr2).mean()
        critic_loss = F.mse_loss(new_values.squeeze(), returns.detach())
        loss = (self.loss_weight['actor'] * actor_loss
                + self.loss_weight['critic'] * critic_loss
                - self.loss_weight['entropy'] / (episode + 1) * entropy.mean())
        return loss

    def learn(self, sample):
        """
        从处理好的SampleData中学习。
        """
        if sample is None or not hasattr(sample, 'actions') or not sample.actions:
            return

        # 直接从sample对象中获取所有需要的数据
        state_seqs = torch.stack(sample.state_seqs)
        action_seqs = torch.stack(sample.action_seqs)
        actions = torch.stack(sample.actions)
        logprobs = torch.stack(sample.logprobs)
        values = torch.stack(sample.values).squeeze() # 添加对values的处理
        rewards = torch.tensor(sample.rewards, dtype=torch.float32) # 添加对rewards的处理
        dones = torch.tensor(sample.dones, dtype=torch.float32) # 添加对dones的处理
        advantages = torch.tensor(sample.advantages, dtype=torch.float32)
        returns = torch.tensor(sample.returns, dtype=torch.float32)

        data_size = len(actions)
        minibatch_size = 512

        for _ in range(self.K_epochs):
            indices = np.arange(data_size)
            np.random.shuffle(indices)

            for start_idx in range(0, data_size, minibatch_size):
                end_idx = min(start_idx + minibatch_size, data_size)
                batch_indices = indices[start_idx:end_idx]

                mb_state_seqs = state_seqs[batch_indices]
                mb_action_seqs = action_seqs[batch_indices]
                mb_actions = actions[batch_indices]
                mb_logprobs = logprobs[batch_indices]
                mb_advantages = advantages[batch_indices]
                mb_returns = returns[batch_indices]

                new_logprobs, new_values, entropy = self.policy.evaluate(
                    mb_state_seqs, mb_action_seqs, mb_actions
                )
                
                loss = self._calc_loss(sample.episode, mb_returns, mb_advantages, mb_logprobs,
                                       new_logprobs, new_values, entropy)

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=2.0)
                self.optimizer.step()

        self.scheduler.step()

    def save_model(self, path=None, id="1"):
        model_file_path = f"{path}/model.ckpt-{str(id)}.pt"
        torch.save({
            "policy": self.policy.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict()
        }, model_file_path)

    def load_model(self, path):
        checkpoint = torch.load(path)
        self.policy.load_state_dict(checkpoint["policy"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.scheduler.load_state_dict(checkpoint["scheduler"])
