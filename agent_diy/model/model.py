import torch
from torch import nn
from torch.distributions import Categorical
import torch.nn.functional as F
import numpy as np
from functools import partial
from kaiwu_agent.utils.common_func import attached, create_cls

LSTMState = create_cls("LSTMState", pi=None, vf=None)
BatchData = create_cls("BatchData", sample=None, mask=None)
SampleData = create_cls("SampleData", rewards=None, dones=None)

class LSTM(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, latent_size, num_layers=1):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.latent_dim = latent_size

        self.lstm = nn.LSTM(
            input_size, 
            hidden_size, 
            num_layers=num_layers, 
            batch_first=True 
            )
        
        self.embedding = nn.Sequential(
            nn.Linear(hidden_size + input_size * 3, latent_size),
            nn.ReLU(),
            nn.Linear(latent_size, output_size)
        )

    def forward(self, input1, input2, hidden_state):
        lstm_output, hidden_state = self.lstm(input2, hidden_state)
        # 拼接输入和LSTM输出
        output = torch.cat((input1, lstm_output), dim=-1)
        # 通过全连接层进行嵌入
        return self.embedding(output), hidden_state
    

class PPO(nn.Module):
    def __init__(
        self,
        feature_dim,
        action_dim,
        lstm_hidden_size,
        n_lstm_layers,
        latent_dim_pi,
        latent_dim_vf
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.action_dim = action_dim
        self.lstm_hidden_size = lstm_hidden_size
        self.n_lstm_layers = n_lstm_layers
        self.latent_dim_pi = latent_dim_pi
        self.latent_dim_vf = latent_dim_vf

        self.feature_net = nn.Linear(feature_dim, feature_dim * 4)

        self.actor = LSTM(
            input_size=feature_dim, 
            output_size=action_dim, 
            hidden_size=lstm_hidden_size, 
            latent_size=latent_dim_pi, 
            num_layers=n_lstm_layers
        )
        self.critic = LSTM(
            input_size=feature_dim, 
            output_size=1, 
            hidden_size=lstm_hidden_size, 
            latent_size=latent_dim_vf, 
            num_layers=n_lstm_layers
        )

        # 对所有网络做正交初始化
        self._orthogonal_init(self.actor)
        self._orthogonal_init(self.critic)

    def _orthogonal_init(self, module: nn.Module, gain=1.0) -> None:
        """网络正交化"""
        for layer in module.modules():
            if isinstance(module, nn.Embedding):
                nn.init.orthogonal_(module.weight, gain=gain)
            elif isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=gain)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LSTM):
                for name, param in module.named_parameters():
                    if 'weight' in name:
                        nn.init.orthogonal_(param, gain=gain)
                    elif 'bias' in name:
                        nn.init.constant_(param, 0)  # LSTM偏置通常初始化为0，或根据门控机制初始化为小正值
            elif isinstance(module, nn.Sequential):
                for sub_module in module:
                    self._orthogonal_init(sub_module, gain=gain)
    

    def _features_extract(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """裁剪分离actor特征与critic特征"""
        # share_features
        # TODO: other features_extractor? 

        pi_features = vf_features = features
        return pi_features, vf_features

    def _features_process(self, features: torch.Tensor, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        """特征处理"""
        # (batch_size, n_envs, feature_dim) 
        # -> (batch_size, n_envs, feature_dim * 4)
        # -> (n_seq, seq_len, n_envs, feature_dim * 4) 
        # -> (n_seq, seq_len, n_envs, feature_dim * 3), (n_seq, seq_len, n_envs, feature_dim)
        # 按学长所说，过FC，再拆分，后拼接
        # TODO: 对于pad 0 的部分，怎么处理
        features = self.feature_net(features)
        features = features.view(-1, seq_len, self.feature_dim * 4)
        features_1 = features[:, :, :self.feature_dim * 3]
        features_2 = features[:, :, self.feature_dim * 3:]

        return features_1, features_2

    def predict_actions(self, features: torch.Tensor, lstm_states: LSTMState, deterministic=False) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """预测动作"""
        # (1, n_envs, feature_dim * 3), (1, n_envs, feature_dim)
        features_1, features_2 = self._features_process(features, 1)
        pi_features, vf_features = self._features_extract(features_2)
        # (1, n_envs, action_dim), lstm_hidden_state
        logits, lstm_states_pi = self.actor(features_1, pi_features, lstm_states.pi)
        # (1, n_envs, 1), lstm_hidden_state
        values, lstm_states_vf = self.critic(features_1, vf_features, lstm_states.vf)

        distribution = Categorical(logits=logits)
        if deterministic:
            # 对应exploit
            actions = distribution.mode # (n_envs, )
        else:
            # 对应predict
            actions = distribution.sample() # (n_envs, )

        log_probs = distribution.log_prob(actions)
        return actions, values, log_probs, LSTMState(pi=lstm_states_pi, vf=lstm_states_vf)

    def evaluate_actions(self, 
                         features: torch.Tensor, 
                         actions: torch.Tensor, 
                         lstm_states: LSTMState, # (batch_size, num_layers, n_envs, hidden)
                         mask: torch.Tensor, 
                         minibatch: int
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # (batch_size, n_envs, feature_dim * 3), (batch_size, n_envs, feature_dim)
        features_1, features_2 = self._features_process(features, minibatch)
        pi_features, vf_features = self._features_extract(features_2)
        # (batch_size, n_envs, action_dim), lstm_hidden_state
        logits, _ = self.actor(features_1, pi_features, lstm_states.pi)
        # (batch_size, n_envs, 1), lstm_hidden_state
        values, _ = self.critic(features_1, vf_features, lstm_states.vf)

        distribution = Categorical(logits=logits)
        log_probs = distribution.log_prob(actions) # (batch_size, n_envs)
        entropy = distribution.entropy() # (batch_size, n_envs)
        return values[mask], log_probs[mask], entropy[mask]

    def predict_values(self, features: torch.Tensor, lstm_states: LSTMState) -> torch.Tensor:
        """预测状态值"""
        # (1, n_envs, feature_dim * 3), (1, n_envs, feature_dim)
        features_1, features_2 = self._features_process(features, 1)
        _, vf_features = self._features_extract(features_2)
        values, _ = self.critic(features_1, vf_features, lstm_states.vf)
        return values

class Model(nn.Module):
    def __init__(
        self,
        feature_dim,                # 特征向量维度
        action_dim,                 # 动作空间维度
        lstm_hidden_size=128,       # LSTM隐藏层
        n_lstm_layers=1,            # LSTM层数
        latent_dim_pi=64,           # Actor策略网络的中间层维度
        latent_dim_vf=64,           # Critic价值网络的中间层维度
        eps_clip=0.1,               # PPO的裁剪参数，限制策略更新的步长
        lr_ppo=3e-4,                # PPO优化器的学习率
        T_max=75,                   # 学习率调度器的调节周期（每多少轮循环一次）
        loss_weight=None,           # Loss权重
        device="auto",              # 训练设备（CPU/GPU），"auto"自动选择
        n_envs=1,                   # 并行环境数量，用于加速数据收集
        K_epochs=10,                # 每次更新使用同一批数据的迭代次数
        minibatch=1024,             # 每个小批量的样本数
        mode="local",         # 训练模式（分布式或本地）
        logger=None                 # 日志记录器，用于记录训练过程和指标
    ):
        super().__init__()
        # PPO
        self.policy = PPO(
            feature_dim = feature_dim,
            action_dim = action_dim,
            lstm_hidden_size = lstm_hidden_size,
            n_lstm_layers = n_lstm_layers,
            latent_dim_pi=latent_dim_pi,
            latent_dim_vf=latent_dim_vf
        )
        self.feature_dim = feature_dim
        self.action_dim = action_dim
        self.latent_dim_pi = latent_dim_pi
        self.latent_dim_vf = latent_dim_vf
        self.lstm_num_layers = n_lstm_layers
        self.lstm_hidden_size = lstm_hidden_size       
        # 优化器
        self.optimizer = torch.optim.AdamW(
            self.policy.parameters(), 
            lr=lr_ppo, 
            weight_decay=1e-4
            )
        self.lr_ppo = lr_ppo
        # 调度器
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, 
            T_max=T_max,  # 半周期
            eta_min=lr_ppo / 100
        )
        self.T_max = T_max
        # Loss
        self.loss_weight = loss_weight
        self.eps_clip = eps_clip
        # Utilities
        self.predict_cnt = 0
        self.device = device
        if logger is not None:
            self.logger_mode = True
            self.logger = logger
        else:
            self.logger_mode = False
        # Buffer
        self.lstm_states = LSTMState(
            pi=(torch.zeros((n_lstm_layers, n_envs, lstm_hidden_size), device=device),
                torch.zeros((n_lstm_layers, n_envs, lstm_hidden_size), device=device)),
            vf=(torch.zeros((n_lstm_layers, n_envs, lstm_hidden_size), device=device),
                torch.zeros((n_lstm_layers, n_envs, lstm_hidden_size), device=device))
        )

        self.other_monitor_data = {
            "loss": []
        }

        self.other_sample_data = {
            "log_probs": None,      # 动作的对数概率
            "values" : None,        # 状态值
            "lstm_states": None,    # LSTM隐状态
        }
        # Env
        self.mode = mode
        self.n_envs = n_envs
        # Train
        self.K_epochs = K_epochs
        self.minibatch = minibatch

    def reset(self):
        self.predict_cnt = 0

        self.lstm_states = LSTMState(
            pi=(torch.zeros((self.lstm_num_layers, self.n_envs, self.lstm_hidden_size), device=self.device),
                torch.zeros((self.lstm_num_layers, self.n_envs, self.lstm_hidden_size), device=self.device)),
            vf=(torch.zeros((self.lstm_num_layers, self.n_envs, self.lstm_hidden_size), device=self.device),
                torch.zeros((self.lstm_num_layers, self.n_envs, self.lstm_hidden_size), device=self.device))
        )

        self.other_monitor_data = {
            "loss": []
        }

        self.other_sample_data = {
            "log_probs": None,          # 动作的对数概率
            "values" : None,            # 状态值
            "hidden_states_pi": None,   # LSTM Pi隐状态 1
            "cell_states_pi": None,     # LSTM Pi隐状态 2
            "hidden_states_vf": None,   # LSTM Vf隐状态 1
            "cell_states_vf": None      # LSTM vf隐状态 2
        }
    
    def _to_torch(self, data: np.ndarray) -> torch.Tensor:
        return torch.tensor(data, device=self.device).float()
    
    def _unpack_samples(self, samples: list[SampleData]) -> dict:
        """将采样数据重组为 BatchData"""
        actual_len = len(samples)
        batch_data = {}
        # 统一转换 为 float32 的 torch.Tensor 
        for attr in [
            "features", 
            "next_features", 
            "actions", 
            "log_probs",
            "rewards", 
            "dones", 
            "values", 
            "advantages", 
            "returns",
            "hidden_states_pi",
            "cell_states_pi",
            "hidden_states_vf",
            "cell_states_vf"
        ]:
            tensor = np.stack([getattr(s, attr) for s in samples], axis=0)
            if actual_len < self.minibatch:
                tensor = np.pad(tensor, ((0, self.minibatch - actual_len), (0, 0)), mode='constant', constant_values=0)
            if attr in ["hidden_states_pi", "cell_states_pi", "hidden_states_vf", "cell_states_vf"]:
                # 特别的 lstm (batch_size, n_lstm_layers, n_envs, lstm_hidden_size) -> (n_lstm_layers, batch_size * n_envs, lstm_hidden_size):
                tensor = tensor.permute(1, 0, 2, 3).reshape(self.lstm_num_layers, -1, self.lstm_hidden_size)
                
            batch_data[attr] = self._to_torch(tensor)
        # Mask
        len_mask = (torch.arange(self.minibatch) < actual_len).unsqueeze(-1)
        done_mask = (~torch.cumsum(batch_data.pop('dones').bool(), dim=0)).unsqueeze(-1) 
        batch_data['mask'] = len_mask & done_mask # (batch_size, n_envs, 1)
 
        return batch_data
    
    def _generate_batch_data(self, list_sample_data):
        """生成批次数据"""
        # (n_steps, n_envs, dim) -> [(batch_size, n_envs, dim), ...]
        
        n_steps = len(list_sample_data)
        for start_idx in range(0, n_steps, self.minibatch):
            end_idx = min(start_idx + self.minibatch, n_steps)
            samples = list_sample_data[start_idx:end_idx] # list / (batch_size, n_envs, dim)
            batch_data = self._unpack_samples(samples) # dict / (batch_size, n_envs, dim)
            yield batch_data

    def _calc_loss(self, values, log_probs, old_log_probs, entropies, advantages, returns):
        ratio = torch.exp(log_probs - old_log_probs)
        policy_loss_1 = advantages * ratio
        policy_loss_2 = advantages * torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip)
        policy_loss = - torch.mean(torch.min(policy_loss_1, policy_loss_2))
        value_loss = torch.mean((returns - values) ** 2)
        entropy_loss = - torch.mean(entropies)

        loss = (self.loss_weight['policy'] * policy_loss
                + self.loss_weight['value'] * value_loss
                + self.loss_weight['entropy'] * entropy_loss)

    def _print_grad_norm(self, total_norms):
        total_grad = sum(total_norms) / len(total_norms) if total_norms else 0.0
        self.logger.info(f"Total Grad: {total_grad:.4f}")
        for name, param in self.policy.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                if grad_norm > 2:  # 只记录高梯度层
                    self.logger.info(f"High grad layer: {name} = {grad_norm:.4f}")


    def learn(self, list_sample_data: list[SampleData]) -> None:
        """训练模型"""
        self.train()
        # TODO: 早停机制
        for epoch in range(self.K_epochs):
            total_norms = []
            for batch_data in self._generate_batch_data(list_sample_data):
                values, log_probs, entropies = self.policy.evaluate_actions(
                    features = self._features_preprocess(batch_data["features"]),
                    actions = batch_data["actions"],
                    lstm_states = LSTMState(
                        pi=(batch_data["hidden_states_pi"], batch_data["cell_states_pi"]),
                        vf=(batch_data["hidden_states_vf"], batch_data["cell_states_vf"])
                    ),
                    mask = batch_data["mask"],
                    minibatch = self.minibatch
                )
                loss = self._calc_loss(
                    values=values,
                    log_probs=log_probs,
                    old_log_probs=batch_data["log_probs"],
                    entropies=entropies,
                    advantages=batch_data["advantages"],
                    returns=batch_data["returns"]
                )
                self.other_monitor_data["loss"].append(loss.item())

                self.optimizer.zero_grad()
                loss.backward()
                total_norm = torch.nn.utils.clip_grad_norm_(
                    self.policy.parameters(), 
                    max_norm=5.0,
                    norm_type=2.0  # L2范数裁剪
                )
                total_norms.append(total_norm)
                self.optimizer.step()
            if self.logger_mode:
                self._print_grad_norm(total_norms)

        self.scheduler.step()
        self.eps_clip *= 0.995

    def _features_preprocess(self, features: np.ndarray) -> torch.Tensor:
        """特征预处理 + 标准化"""
        features = self._to_torch(features)
        features = torch.clamp(features, min=-10.0, max=10.0)
        features = torch.nan_to_num(features, nan=0.0, posinf=10.0, neginf=-10.0)

        features = (features - features.mean()) / (features.std() + 1e-6)
        return features

    def _actions_preprocess(self, actions: np.ndarray) -> torch.Tensor:
        """动作预处理"""
        # (n_envs, ) -> (n_envs, action_dim)
        actions = self._to_torch(actions)
        actions = F.one_hot(actions, num_classes=self.action_dim).float()  # 转为one-hot向量
        return actions

    def predict(self, features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """在训练模式下预测动作"""
        self.predict_cnt += 1
        features = self._features_preprocess(features).unsqueeze(0) # (1, n_envs, feature_dim)
        with torch.no_grad():
            actions, values, log_probs, self.lstm_states = self.policy.predict_actions(
                features, 
                self.lstm_states, 
                deterministic=False
            )
        # 约定储存进buffer的都是 np.ndarray
        self.other_sample_data["log_probs"] = log_probs.numpy()
        self.other_sample_data["values"] = values.numpy()
        self.other_sample_data["hidden_states_pi"] = self.lstm_states.pi[0].numpy()
        self.other_sample_data["cell_states_pi"] = self.lstm_states.pi[1].numpy()
        self.other_sample_data["hidden_states_vf"] = self.lstm_states.vf[0].numpy()
        self.other_sample_data["cell_states_vf"] = self.lstm_states.vf[1].numpy()

        return actions.squeeze(0).detach(), log_probs.squeeze(0).detach()  # (n_envs, ), (n_envs, )

    def exploit(self, features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """在推理模式下预测动作"""
        features = self._features_preprocess(features).unsqueeze(0)
        with torch.no_grad():
            actions, values, log_probs, self.lstm_states = self.policy.predict_actions(
                features, 
                self.lstm_states, 
                deterministic=True
            )

        return actions.squeeze(0).detach(), log_probs.squeeze(0).detach()  # (n_envs, action_dim), (n_envs, action_dim)

    def handle_timeout(self, truncateds: np.ndarray, rewards: np.ndarray, features: np.ndarray) -> np.ndarray:
        # TODO
        return rewards

    def compute_values(self, features: np.ndarray, lstm_hidden_state: LSTMState) -> np.ndarray:
        """计算返回和优势 中的 values"""
        with torch.no_grad():
            features = self._features_preprocess(features) # (total_steps, n_envs, feature_dim)
            values = self.policy.predict_values(features, lstm_hidden_state)  # (total_steps, n_envs, 1)

        return values.squeeze(-1).numpy() # (total_steps, n_envs)

    def get_other_sample_data(self):
        """基于模型实现, 补充其它需要采样的数据"""
        return self.other_sample_data
    
    def get_other_monitor_data(self):
        """基于监控需求, 补充其它需要监控的数据"""
        return self.other_monitor_data
    
    def save_model(self, path=None, id="1"):
        model_file_path = f"{path}/model.ckpt-{str(id)}.pt"
        torch.save({
            "policy": self.policy.state_dict()
        }, model_file_path)

    def load_model(self, path=None, id="1"):
        model_file_path = f"{path}/model.ckpt-{str(id)}.pt"
        checkpoint = torch.load(model_file_path, map_location=self.device)
        self.policy.load_state_dict(checkpoint["policy"])
        self.reset()