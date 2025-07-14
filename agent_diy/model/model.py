import torch
from torch import nn
from torch.distributions import Categorical
import torch.nn.functional as F
import numpy as np
from functools import partial
from kaiwu_agent.utils.common_func import attached, create_cls

SampleData = create_cls("SampleData", rewards=None, dones=None)
LSTMState = create_cls("LSTMState", pi=None, vf=None)
BufferData = create_cls(
    "BufferData", 
    features=None,
    actions=None,
    values=None,
    log_probs=None,
    advantages=None,
    returns=None,
    lstm_states=None,
    episode_starts=None,
    mask=None
    )


def pad(
    seq_start_indices,
    seq_end_indices,
    device,
    tensor,
    padding_value: float = 0.0,
    ):
    # Create sequences given start and end
    seq = [torch.tensor(tensor[start : end + 1], device=device) for start, end in zip(seq_start_indices, seq_end_indices)]
    return torch.nn.utils.rnn.pad_sequence(seq, batch_first=True, padding_value=padding_value)


def pad_and_flatten(
    seq_start_indices,
    seq_end_indices,
    device,
    tensor,
    padding_value: float = 0.0,
    ):
    return pad(seq_start_indices, seq_end_indices, device, tensor, padding_value).flatten()


class Buffer:
    def __init__(
        self,
        buffer_size,
        feature_shape,
        action_dim,
        hidden_state_shape,
        device="auto",
        n_envs=1,
        gamma=0.99,
        gae_lambda=1
    ):
        # 环境数据
        self.buffer_size = max(buffer_size // n_envs, 1)
        self.n_envs = n_envs
        self.feature_shape = feature_shape
        self.action_dim = action_dim
        self.device = device
        # 算法参数
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        # LSTM
        self.hidden_state_shape = hidden_state_shape
        self.seq_start_indices, self.seq_end_indices = None, None

    def reset(self):
        # 状态存储
        self.features = np.zeros((self.buffer_size, self.n_envs, *self.feature_shape))
        self.actions = np.zeros((self.buffer_size, self.n_envs, self.action_dim))
        self.rewards = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.episode_starts = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.values = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.log_probs = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.advantages = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        # buffer数据
        self.pos = 0
        self.full = False
        # LSTM
        # self.env_change = np.zeros(self.buffer_size * self.n_envs).reshape(self.buffer_size, self.n_envs) # 0-1向量 记录env转换
        self.hidden_states_pi = np.zeros(self.hidden_state_shape, dtype=np.float32)
        self.cell_states_pi = np.zeros(self.hidden_state_shape, dtype=np.float32)
        self.hidden_states_vf = np.zeros(self.hidden_state_shape, dtype=np.float32)
        self.cell_states_vf = np.zeros(self.hidden_state_shape, dtype=np.float32)

    def add(
        self,
        feature,
        action,
        reward,
        episode_start,
        value,
        log_prob,
        lstm_states: LSTMState
    ):
        feature = feature.reshape((self.n_envs, *self.feature_shape))
        action = action.reshape((self.n_envs, self.action_dim))

        self.features[self.pos] = np.array(feature)
        self.actions[self.pos] = np.array(action)
        self.rewards[self.pos] = np.array(reward)
        self.episode_starts[self.pos] = np.array(episode_start)
        self.values[self.pos] = value.clone().cpu().numpy().flatten()
        self.log_probs[self.pos] = log_prob.clone().cpu().numpy()

        self.hidden_states_pi[self.pos] = np.array(lstm_states.pi[0].cpu().numpy())
        self.cell_states_pi[self.pos] = np.array(lstm_states.pi[1].cpu().numpy())
        self.hidden_states_vf[self.pos] = np.array(lstm_states.vf[0].cpu().numpy())
        self.cell_states_vf[self.pos] = np.array(lstm_states.vf[1].cpu().numpy())

        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True

    def compute_returns_and_advantage(self, last_values, dones):
        last_values = last_values.clone().cpu().numpy().flatten()
        last_gae_lam = 0
        for step in reversed(range(self.buffer_size)):
            if step == self.buffer_size - 1:
                next_non_terminal = 1.0 - dones.astype(np.float32)
                next_values = last_values
            else:
                next_non_terminal = 1.0 - self.episode_starts[step + 1]
                next_values = self.values[step + 1]
            delta = self.rewards[step] + self.gamma * next_values * next_non_terminal - self.values[step]
            last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
            self.advantages[step] = last_gae_lam
        self.returns = self.advantages + self.values

    def get_batch_data(self, minibatch):
        split_index = np.random.randint(self.buffer_size * self.n_envs)
        indices = np.arange(self.buffer_size * self.n_envs)
        indices = np.concatenate((indices[split_index:], indices[:split_index]))

        self._get_ready()

        for start_idx in range(0, self.buffer_size * self.n_envs, minibatch):
            yield self._get_samples(indices[start_idx : start_idx + minibatch])

    def _get_ready(self):
        for tensor in ["hidden_states_pi", "cell_states_pi", "hidden_states_vf", "cell_states_vf"]:
            self.__dict__[tensor] = self.__dict__[tensor].swapaxes(1, 2)

        for tensor in [
                "features",
                "actions",
                "values",
                "log_probs",
                "advantages",
                "returns",
                "hidden_states_pi",
                "cell_states_pi",
                "hidden_states_vf",
                "cell_states_vf",
                "episode_starts"
            ]:
                self.__dict__[tensor] = self._swap_and_flatten(self.__dict__[tensor])

    def _to_torch(data):
        return torch.tensor(array, device=self.device)

    def _get_sample(self, batch_indices):
        self.seq_start_indices, self.pad, self.pad_and_flatten = self._create_sequencers(self.episode_starts[batch_indices])

        n_seq = len(self.seq_start_indices)
        max_length = self.pad(self.actions[batch_indices]).shape[1]
        padded_batch_size = n_seq * max_length

        lstm_states_pi = (
            # (n_envs * n_steps, n_layers, dim) -> (n_layers, n_seq, dim)
            self.hidden_states_pi[batch_indices][self.seq_start_indices].swapaxes(0, 1),
            self.cell_states_pi[batch_indices][self.seq_start_indices].swapaxes(0, 1),
        )
        lstm_states_vf = (
            # (n_envs * n_steps, n_layers, dim) -> (n_layers, n_seq, dim)
            self.hidden_states_vf[batch_indices][self.seq_start_indices].swapaxes(0, 1),
            self.cell_states_vf[batch_indices][self.seq_start_indices].swapaxes(0, 1),
        )
        lstm_states_pi = (self._to_torch(lstm_states_pi[0]).contiguous(), self._to_torch(lstm_states_pi[1]).contiguous())
        lstm_states_vf = (self._to_torch(lstm_states_vf[0]).contiguous(), self._to_torch(lstm_states_vf[1]).contiguous())

        return BufferData(
            # (batch_size, features_dim) -> (n_seq, max_length, features_dim) -> (n_seq * max_length, features_dim)
            features=self.pad(self.features[batch_indices]).reshape((padded_batch_size, *self.features_shape)),
            actions=self.pad(self.actions[batch_indices]).reshape((padded_batch_size,) + self.actions.shape[1:]),
            values=self.pad_and_flatten(self.values[batch_indices]),
            log_probs=self.pad_and_flatten(self.log_probs[batch_indices]),
            advantages=self.pad_and_flatten(self.advantages[batch_indices]),
            returns=self.pad_and_flatten(self.returns[batch_indices]),
            lstm_states=LSTMStates(pi=lstm_states_pi, vf=lstm_states_vf),
            episode_starts=self.pad_and_flatten(self.episode_starts[batch_indices]),
            mask=self.pad_and_flatten(np.ones_like(self.returns[batch_indices])),
        )

    def _swap_and_flatten(self, arr):
        """
        Swap and then flatten axes 0 (buffer_size) and 1 (n_envs)
        to convert shape from [n_steps, n_envs, ...] (when ... is the shape of the features)
        to [n_steps * n_envs, ...] (which maintain the order)
        """
        shape = arr.shape
        if len(shape) < 3:
            shape = (*shape, 1)
        return arr.swapaxes(0, 1).reshape(shape[0] * shape[1], *shape[2:])

    def _create_sequencers(self, episode_starts):
        seq_start = episode_starts.flatten()
        seq_start[0] = True
        seq_start_indices = np.where(seq_start == True)[0]
        seq_end_indices = np.concatenate([(seq_start_indices - 1)[1:], np.array([len(episode_starts)])])

        local_pad = partial(pad, seq_start_indices, seq_end_indices, self.device)
        local_pad_and_flatten = partial(pad_and_flatten, seq_start_indices, seq_end_indices, self.device)
        return seq_start_indices, local_pad, local_pad_and_flatten

class LSTMPPO(nn.Module):
    def __init__(
        feature_dim,
        action_dim,
        lstm_hidden_size,
        n_lstm_layers,
    ):
        self.feature_dim = feature_dim
        self.action_dim = action_dim
        self.lstm_hidden_size = lstm_hidden_size
        self.n_lstm_layers = n_lstm_layers
        self.latent_dim_pi = latent_dim_pi
        self.latent_dim_vf = latent_dim_vf
    
        self.policy_net = nn.Sequential(
            nn.Linear(feature_dim, latent_dim_pi),
            nn.ReLU(),
            nn.Linear(latent_dim_pi, latent_dim_pi),
        )
        self.value_net = nn.Sequential(
            nn.Linear(feature_dim, latent_dim_pi),
            nn.ReLU(),
            nn.Linear(latent_dim_pi, latent_dim_pi),
        )
        
        self.actor_net = nn.Linear(self.latent_dim_pi, self.action_dim)
        self.critic_net = nn.Linear(self.mlp_extractor.latent_dim_vf, 1)
        self.lstm_actor = nn.LSTM(
            self.features_dim,
            lstm_hidden_size,
            num_layers=n_lstm_layers,
        )
        self.lstm_critic = nn.LSTM(
            self.features_dim,
            lstm_hidden_size,
            num_layers=n_lstm_layers,
            )

        self.lstm_hidden_state_shape = (n_lstm_layers, 1, lstm_hidden_size)

    def forward_actor():
        return self.policy_net(features)

    def forward_critic():
        return self.value_net(features)
    
    def forward():
        pi_features = vf_features = feature
        latent_pi, lstm_states_pi = self._process_sequence(pi_features, lstm_states.pi, episode_starts, self.lstm_actor)
        latent_vf, lstm_states_vf = self._process_sequence(vf_features, lstm_states.vf, episode_starts, self.lstm_critic)

        latent_pi = self.forward_actor(latent_pi)
        latent_vf = self.forward_critic(latent_vf)

        values = self.critic_net(latent_vf)
        mean_actions = self.actor_net(latent_pi)
        distribution = Categorical(logits=mean_actions)

        if deterministic:
            action = distribution.mode()
        else:
            action = distribution.sample()
        
        log_prob = distribution.log_prob(actions)
        return actions, values, log_prob, LSTMState(pi=lstm_states_pi, vf=lstm_states_vf)

    def _process_sequence(self, feature, lstm_states, episode_starts, lstm):
        n_seq = lstm_states[0].shape[1]
        features_sequence = features.reshape((n_seq, -1, lstm.input_size)).swapaxes(0, 1)
        episode_starts = episode_starts.reshape((n_seq, -1)).swapaxes(0, 1)
        if torch.all(episode_starts == 0.0):
            lstm_output, lstm_states = lstm(features_sequence, lstm_states)
            lstm_output = torch.flatten(lstm_output.transpose(0, 1), start_dim=0, end_dim=1)
            return lstm_output, lstm_states

        lstm_output = []
        # Iterate over the sequence
        for features, episode_start in zip(features_sequence, episode_starts):
            hidden, lstm_states = lstm(
                features.unsqueeze(dim=0),
                (
                    (1.0 - episode_start).view(1, n_seq, 1) * lstm_states[0],
                    (1.0 - episode_start).view(1, n_seq, 1) * lstm_states[1],
                ),
            )
            lstm_output += [hidden]
        # Sequence to batch
        # (sequence length, n_seq, lstm_out_dim) -> (batch_size, lstm_out_dim)
        lstm_output = torch.flatten(torch.cat(lstm_output).transpose(0, 1), start_dim=0, end_dim=1)
        return lstm_output, lstm_states

    def set_training_mode(self, mode):
        self.train(mode)

    def evaluate_actions():
        pi_features, vf_features = self._extract_features(feature)
        latent_pi, _ = self._process_sequence(pi_features, lstm_states.pi, episode_starts, self.lstm_actor)
        latent_vf, _ = self._process_sequence(vf_features, lstm_states.vf, episode_starts, self.lstm_critic)
        
        latent_pi = self.forward_actor(latent_pi)
        latent_vf = self.forward_critic(latent_vf)

        values = self.critic_net(latent_vf)
        mean_actions = self.actor_net(latent_pi)
        distribution = Categorical(logits=mean_actions)
        log_prob = distribution.log_prob(actions)
        return values, log_prob, distribution.entropy()

    def predict_values():
        latent_vf, _ = self._process_sequence(features, lstm_states, episode_starts, self.lstm_critic)
        values = self.forward_critic(latent_vf)
        return self.critic_net(latent_vf)

    def predict():
        return self.forward()

    def exploit():
        return self.forward()

class Model(nn.Module):
    def __init__():
        # 训练工具
        self.policy = LSTMPPO()
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr_ppo)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=step_size, gamma=lr_scheduler)
        self.buffer = None
        # 状态存储
        self.last_feature = None
        self.last_action = None
        self.last_value = None
        self.last_log_prob = None
        self.last_lstm_states = None
        # 训练参数
        self.device = device
        ## 网络相关
        self.buffer_size = buffer_size
        self.lstm_num_layers = lstm_num_layers
        self.lstm_hidden_size = lstm_hidden_size
        ## 环境相关
        self.n_envs = n_envs
        ## 梯度相关
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.eps_clip = eps_clip
        self.step_size = step_size
        self.lr_scheduler = lr_scheduler
        self.lr_ppo = lr_ppo
        ## 批次相关
        self.K_epochs = K_epochs
        self.minibatch = minibatch
        # TODO: logger
        if logger is not None:
            self.logger_mode = True
            self.logger = logger
        else:
            self.logger_mode = False

    def reset(self):
        single_hidden_state_shape = (self.lstm_num_layers, self.n_envs, self.lstm_hidden_size)
        self.last_lstm_states = LSTMState(
            (
                th.zeros(single_hidden_state_shape, device=self.device),
                th.zeros(single_hidden_state_shape, device=self.device),
            ),
            (
                th.zeros(single_hidden_state_shape, device=self.device),
                th.zeros(single_hidden_state_shape, device=self.device),
            ),
        )
        hidden_state_buffer_shape = (self.buffer_size, self.lstm_num_layers, self.n_envs, self.lstm_hidden_size)
        self.buffer = Buffer(
            buffer_size = self.buffer_size,
            feature_shape = self.feature_shape,
            action_dim = self.action_dim,
            hidden_state_shape = hidden_state_buffer_shape,
            device = self.device,
            n_envs = self.n_envs,
            gamma = self.gamma,
            gae_lambda = self.gae_lambda
        )
        self.buffer.reset()

    def predict():
        pass

    def exploit():
        pass

    def learn(self):
        self.policy.set_training_mode(True)
        self.scheduler.step()

        for epoch in range(self.K_epochs):
            for rollout_data in self.buffer.get_batch_data(self.minibatch):
                mask = rollout_data.mask > 1e-8
                values, log_probs, entropies = self.policy.evaluate_actions(
                    rollout_data.features,
                    rollout_data.actions,
                    rollout_data.lstm_states,
                    rollout_data.episode_starts,
                )

                advantages = rollout_data.advantages
                advantages = (advantages - advantages[mask].mean()) / (advantages[mask].std() + 1e-8)

                ratio = torch.exp(log_probs - rollout_data.log_probs)
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip)
                policy_loss = - torch.mean(torch.min(policy_loss_1, policy_loss_2)[mask])

                value_loss = torch.mean(((rollout_data.returns - values) ** 2)[mask])

                entropy_loss = - torch.mean(entropies[mask])

                loss = (self.loss_weight['policy'] * policy_loss 
                        + self.loss_weight['value'] * value_loss
                        + self.loss_weight['entropy'] * entropy_loss)

                # TODO: 早停机制
                # XXXXX

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.policy.parameters(), 
                    max_norm=5.0,
                    norm_type=2.0  # L2范数裁剪
                )
                self.optimizer.step()

    def collect_rollouts(self, sample):
        # TODO: SDE

        self.buffer.add(
            feature = self.last_feature,
            action = self.last_action,
            reward = sample.reward,
            episode_start = sample.done,
            value = self.last_value,
            log_prob = self.last_log_prob,
            lstm_states = self.last_lstm_states
        )
        
    def save_model(self, path=None, id="1"):
        model_file_path = f"{path}/model.ckpt-{str(id)}.pt"
        torch.save({
            "policy": self.policy.state_dict()
        }, model_file_path)

    def load_model(self, path=None, id="1"):
        model_file_path = f"{path}/model.ckpt-{str(id)}.pt"
        checkpoint = torch.load(model_file_path)
        self.policy.load_state_dict(checkpoint["policy"])
        self.reset()