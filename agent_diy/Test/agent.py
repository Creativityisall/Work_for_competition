import numpy as np

from conf import Config
from utils import create_cls
from model import Model

ObsData = create_cls("ObsData", feature=None, legal_actions=None)
ActData = create_cls("ActData", action=None, prob=None)

class Agent:
    def __init__(self, agent_type="player", device=None, logger=None, monitor=None) -> None:
        self.model = Model(
            feature_dim = Config.FEATURE_DIM,
            action_dim = Config.ACTION_DIM,
            lstm_hidden_size = Config.LSTM_HIDDEN_SIZE,
            n_lstm_layers = Config.N_LSTM_LAYERS,
            latent_dim_pi = Config.LATENT_DIM_PI,
            latent_dim_vf = Config.LATENT_DIM_VF,
            gamma = Config.GAMMA,
            gae_lambda = Config.GAE_LAMBDA,
            eps_clip = Config.EPSILON,
            lr_ppo = Config.LR_PPO,
            T_max=Config.T_MAX,
            loss_weight=Config.LOSS_WEIGHT,
            device = device,
            buffer_size = Config.BUFFER_SIZE,
            n_envs = Config.N_ENVS,
            K_epochs = Config.K_EPOCHS,
            minibatch = Config.MINIBATCH,
            logger=logger
        )
        self.logger = logger
        self.device = device
        self.monitor = monitor
        self.update_interval = Config.UPDATE

    def reset(self):
        self.model.reset()

    def _features_extract(self, list_obs_data: list[ObsData]) -> tuple[np.ndarray, np.ndarray]:
        features = []
        legal_actions = []
        for obs_data in list_obs_data:
            features.append(obs_data.feature)
            legal_actions.append(obs_data.legal_actions)

        return np.array(features), np.array(legal_actions)

    def predict(self, list_obs_data: list[ObsData]) -> list[ActData]:
        features, legal_actions = self._features_extract(list_obs_data)

        list_act_data = []
        actions, log_probs = self.model.predict(features) # (n_envs, action_dim)
        for action, log_prob in zip(actions, log_probs):
            act_data = ActData(action=action.to('cpu'), prob=np.exp(log_prob.to('cpu')))
            # self.logger.info(f"{act_data.action} - {act_data.prob}")
            list_act_data.append(act_data)
        return list_act_data

    def exploit(self, list_obs_data):
        obs, extra_info = list_obs_data["obs"], list_obs_data["extra_info"]
        list_obs_data=self.observation_process(list_obs=[obs], list_extra_info=[extra_info])
        features, legal_actions = self._features_extract(list_obs_data)

        actions, log_probs = self.model.exploit(features) # (n_envs, action_dim)
        list_act_data = []
        for action, log_prob in zip(actions, log_probs):
            act_data = ActData(action=action.to('cpu'), prob=np.exp(log_prob.to('cpu')))
            self.logger.info(f"{act_data.action} - {act_data.prob}")
            list_act_data.append(act_data)

        actions = self.action_process(list_act_data=list_act_data)
        return actions[0] #TODO: 分布式

    def learn(self):
        self.model.learn()
        self.model.buffer.reset()

    def action_process(self, list_act_data: list[ActData]) -> list[int]:
        actions = []
        for act_data in list_act_data:
            actions.append(act_data.action.item())
        return actions

    def collect(self, sample_data, list_obs_data) -> None:
        """采集环境反馈"""
        features, legal_actions = self._features_extract(list_obs_data)
        self.model.collect_rollouts(sample_data, features)

    def set_feature(self, list_obs_data):
        features, legal_actions = self._features_extract(list_obs_data)
        self.model.last_features = features

    def handle_timeout(self, truncateds, rewards, list_obs_data):
        features = []
        for obs_data in list_obs_data:
            features.append(obs_data.feature)

        return self.model.handle_timeout(truncateds, rewards, features)

    def compute_returns_and_advantage(self):
        """Compute value for the last timestep"""
        self.model.compute_returns_and_advantage()

    def collect_full(self) -> bool:
        """判断采样数据是否足够"""
        return (self.model.buffer.add_cnt > self.update_interval and self.model.buffer.full)

    def save_model(self, path=None, id="1"):
        self.model.save_model(path, id)

    def load_model(self, path=None, id="1"):
        self.model.load_model(path, id)

    def _single_observation_process(self, obs, extra_info):
        # CartPole的动作始终是[0, 1]
        legal_actions = [0, 1]
        # 特征就是环境的观测值
        feature = obs
        return ObsData(feature=feature, legal_actions=legal_actions)

    def observation_process(self, list_obs, list_extra_info):
        list_obs_data = []
        for obs, extra_info in zip(list_obs, list_extra_info):
            obs_data = self._single_observation_process(obs, extra_info)
            list_obs_data.append(obs_data)

        return list_obs_data