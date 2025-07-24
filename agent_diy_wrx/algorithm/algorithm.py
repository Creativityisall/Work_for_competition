#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2025 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


import numpy as np
import torch



from agent_diy.conf.conf import Config
from agent_diy.model.model import FeatureEncoderModel, LstmModel, PpoModel

from agent_diy.agent import ObsData



class LstmPpoAlgorithm:
    # TODO GTL 增加更多算法相关的参数。注意别漏技巧，比如学习率退火。
    # 在这里设置的，都是类属性，无需实例化类对象，只需 from .. import LstmPpoAlgorithm 即可调用 LstmPpoAlgorithm.gamma 等。
    gamma = Config.gamma
    # More ...
    

    def __init__(self, device=None, logger=None, monitor=None):
        # NOTE 算法会一次性从 Config 里加载所有任务需要的参数到自己的属性里。
        # NOTE Config 里应该只有最基本的参数配置。组合参数应该在 LstmPpoAlgorithm 类的 __init__ 方法里计算。
        self.logger = logger
        self.monitor = monitor
        self.device = device


        self.feature_dim = Config.feature_dim
        self.feature_encoder_dim = Config.feature_encoded_dim
        self.action_dim = Config.action_dim

        ############ Feature Encoder Network Parameters ############
        self.feature_encoder_input_dim = self.feature_dim
        self.feature_encoder_output_dim = self.feature_encoder_dim


        ############ LSTM shape configuration ############  
        """
        input: (seq_len, b, feature_encoded_dim // 4, )
        hidden_state (h and c): (L, b, H) 

        output: (seq_len, b, feature_encoded_dim // 4, )
        """

        self.lstm_pi_latent_dim = Config.feature_encoded_dim // 4 
        self.lstm_pi_hidden_dim = Config.lstm_pi_hidden_dim
        self.lstm_pi_num_layers = Config.lstm_pi_num_layers
        self.lstm_pi_seq_len = Config.lstm_pi_seq_len # =1

        self.lstm_vf_latent_dim = Config.feature_encoded_dim // 4
        self.lstm_vf_hidden_dim = Config.lstm_vf_hidden_dim
        self.lstm_vf_num_layers = Config.lstm_vf_num_layers
        self.lstm_vf_seq_len = Config.lstm_vf_seq_len # =1

        ############ Actor and Critic Network shape configuration, both of whose input_dim = feature_encoded_dim * (1/4 + 3/4) ############
        self.policy_input_dim = Config.feature_encoded_dim 
        self.value_input_dim = Config.feature_encoded_dim
                                    

        # Network models
        self.feature_encoder = FeatureEncoderModel(
            device=self.device,
            logger=self.logger,
            monitor=self.monitor,

            input_dim=self.feature_encoder_input_dim,
            output_dim=self.feature_encoder_output_dim
        )



        self.lstm_model = LstmModel(
            device=self.device,
            logger=self.logger,
            monitor=self.monitor,

            pi_input_size=self.lstm_pi_latent_dim,
            pi_hidden_size=self.lstm_pi_hidden_dim,
            pi_seq_len=self.lstm_pi_seq_len,
            pi_num_layers=self.lstm_pi_num_layers,

            vf_input_size=self.lstm_vf_latent_dim,
            vf_hidden_size=self.lstm_vf_hidden_dim,
            vf_seq_len=self.lstm_vf_seq_len,
            vf_num_layers=self.lstm_vf_num_layers,
        )

        self.ppo_model = PpoModel(
            device=self.device,
            logger=self.logger,
            monitor=self.monitor, 

            policy_net_input_dim=self.feature_encoder_dim,
            policy_net_output_dim=self.action_dim,
            value_net_input_dim=self.feature_encoder_dim,
            value_net_output_dim=1,
            
            # 不需要向网络里传递 PPO algo-related parameters，因为所有算法行为都在 LstmPpoAlgorithm 类里实现。
        )

    

    # def get_action_and_value(self, x, action=None): ...    
    def forward_a_step(self, list_obs_data, list_action=None, deterministic=False):
        """
        LstmPpoAlgorithm 算法类提供的前向传播方法。地位和 get_action_and_value 一样。相当于遍历了一遍网络。
        该方法是 Agent 和 LstmPpoAlgorithm 的交互接口；也是类内 learn 时要用的方法。
        - 调用它的 Agent 方法有：predict, exploit；
        - 同时 LstmPpoAlgorithm 内部的 learn 也会调用它。

        TODO xrq 更新！！加一层特征提取网络，用后 1/4 喂给 lstm，再和前 3/4 拼接起来，交给 value 和 policy 网络。
        NOTE 可以为 actor 和 critic 都设计一个特征提取网络，也可以共用一套。
        NOTE algo.forward_a_step(..) 会更麻烦一点，其他不影响（learn的阶段，还是从头过一遍所有网，只容忍 hidden_state 是 off-policy的）  

        输入：
        - 参数1：一个 ObsData 类型元素的列表，其中元素类型为 ObsData，存储走这步前的状态：
            - 原始局部观测，经过提取后的特征 feature
            - Value 和 Policy 网络前置 LSTM 网络当前的隐藏态 lstm_state_pi 和 lstm_state_vf
        - 参数2：是否指定动作
            - 指定：更新网络时调用，计算并返回网络更新后的 actions（无用）, newlogprobs, entropies, newvalues，后三者计算 loss 时有用。
            - 不指定：预测动作时调用。此时参数3起效，若 predict 调用则为收集数据阶段，使用随机采样；若 exploit 调用则为评估阶段，使用贪婪采样。
        
        
        输出：actions, newlogprobs, entropys, values （都是 tensor 类型）

        本函数实现中需要 tensor 化列表形式的输入参数，因为底层是神经网络执行 forward，需要调整数据位批处理的形状。这样调用此方法者按照参数名提示传递列表即可。
        """



        """
        1. Forward pass through Feature Encoder Network
        """
        features_list = [obs_data.feature for obs_data in list_obs_data]
        features_tensor = torch.tensor(features_list, dtype=torch.float32, device=self.device).contiguous()
        features_encoded = self.feature_encoder.forward(features_tensor)  # (batch_size, feature_dim) -> (batch_size, feature_encoded_dim)


        """
        2. 1/4 feature_encoded vector - Forward pass through LSTM networks

        通过 LSTM 网络前向传播，得到 latent_pf 和 latent_vf。
        它们的形状分别为 (seq_len=1, batch_size, pi_hidden_size) 和 (seq_len=1, batch_size, vf_hidden_size)。

        步骤：
        (a). 将 lstm_state_pi 和 lstm_state_vf 转换为 tensor
        (b). 截断输入，保留后 1/4 （学习过程中自动化后半截留给“记忆”）
        (c). 调用 LSTM 网络的 forward 方法，得到 latent_pf 和 latent_v
        (d). 更新当前的 LSTM 隐藏状态（如果需要）
        
        NOTE LSTM 网络本身可以处理任意长度的序列，但本框架设置 seq_len=1 （并且因此简化了很多实现，如padding等都用不到）即每次 forward_a_step 只处理一个时间步的数据。
        后期如果需要，可以处理更长的序列：将 seq_len 设置为更大的值，并在输入数据中提供相应的序列长度。
        不过，将处理数据的序列长度为1，LSTM 网络其实已能够捕捉到状态之间的依赖关系，因为它们的隐藏状态会在每次 forward_a_step 调用中更新。
        """
        ############ (a) 提取隐藏态 ############
        # lstm_states_* 是 [(h_*, c_*), ...]，每个 h_* 和 c_* 的形状都是 (*_num_layers, 1, *_hidden_size)

        lstm_states_pi_list = [obs_data.lstm_state_pi for obs_data in list_obs_data]
        lstm_states_vf_list = [obs_data.lstm_state_vf for obs_data in list_obs_data]
        
        h_pi_list = [torch.tensor(state[0], dtype=torch.float32) for state in lstm_states_pi_list]
        c_pi_list = [torch.tensor(state[1], dtype=torch.float32) for state in lstm_states_pi_list]
        hs_pi = torch.cat(h_pi_list, dim=1).contiguous() # (pi_num_layers, batch_size, pi_hidden_size)
        cs_pi = torch.cat(c_pi_list, dim=1).contiguous()
        lstm_states_pi_tensor = (hs_pi, cs_pi)

        h_vf_list = [torch.tensor(state[0], dtype=torch.float32) for state in lstm_states_vf_list]
        c_vf_list = [torch.tensor(state[1], dtype=torch.float32) for state in lstm_states_vf_list]
        hs_vf = torch.cat(h_vf_list, dim=1).contiguous() # 沿 batch_size 维度拼接（dim=1）
        cs_vf = torch.cat(c_vf_list, dim=1).contiguous()
        lstm_states_vf_tensor = (hs_vf, cs_vf)

        ############ (b) 截断输入，保留后 1/4 ############
        # feature_encoded_tensor 的形状是 (batch_size, feature_encoded_dim)，需要转换为 (seq_len=1, batch_size, feature_encoded_dim // 4)，
        # 以与 LSTM 网络的输入形状匹配。
        # NOTE 如果 seq_len 不是 1，则需要大动这里的逻辑。
        features_encoded = features_encoded.unsqueeze(0)  # (1, batch_size, feature_encoded_dim)
        
        # 切分特征编码向量：前 3/4 和后 1/4
        split_point = self.feature_encoder_dim * 3 // 4
        features_encoded_major = features_encoded[:, :, :split_point]  # 前 3/4，(1, batch_size, feature_encoded_dim * 3 // 4)
        features_encoded_minor = features_encoded[:, :, split_point:]  # 后 1/4，用于 LSTM
        
        # 重新调整形状以匹配 LSTM 输入
        features_encoded_minor = features_encoded_minor.permute(1, 0, 2).contiguous()  # (seq_len=1, batch_size, feature_encoded_dim // 4)
        



        ############ (c) forward pass through LSTM networksv ############
        # latent_* : (seq_len=1, batch_size, lstm_*_output_size=lstm_*_input_size = feature_encoded_dim // 4)
        latent_pi, next_hidden_state_pi = self.lstm_model.pi_forward(
            features_encoded_minor,
            lstm_states_pi_tensor
        )

        latent_vf, next_hidden_state_vf = self.lstm_model.vf_forward(
            features_encoded_minor,
            lstm_states_vf_tensor
        )

        ############ (d) Update the current hidden state of LSTM networks, if necessary. ############
        # 若 batch_size=1，说明是 rollout 阶段的 forward_a_step，则更新 self.algo.lstm_model 的当前隐藏状态
        # 否则，是 learn 阶段的 forward_a_step，只需要得到 _, new_logprobs, entropies, new_values 即可，不需要更新当前隐藏状态（完全依靠 list_obs_data 中存储的features 和 隐藏层数据向前传播）
        if features_tensor.shape[1] == 1:
            # 更新当前的 LSTM 隐藏状态
            self.lstm_model.current_hidden_state_pi = next_hidden_state_pi
            self.lstm_model.current_hidden_state_vf = next_hidden_state_vf
        else:
            pass

        
        
        """
        3. 把上一层的 latent 输出转为 actor 和 critic 网络的输入并向前传播，得到四个张量作为输出：actions, logprobs, entropies, values。

        重新拼接 latent 和前 3/4 的 feature_encoded 向量，作为下一层 Policy 和 Value 网络的输入。
        """
        # 展平 latent 张量，从 (seq_len, batch_size, ) 到 (batch_size, feature_encoded_dim // 4)
        latent_pi_flattened = latent_pi.view(latent_pi.size(1), -1)  # (batch_size, feature_encoded_dim // 4)
        latent_vf_flattened = latent_vf.view(latent_vf.size(1), -1)  # (batch_size, feature_encoded_dim // 4)
        # 展平 feature_encoded_major 到 (batch_size, 3 * feature_encoded_dim // 4)
        features_encoded_major_flattened = features_encoded_major.view(features_encoded_major.size(1), -1)

        policy_input = torch.cat([features_encoded_major_flattened, latent_pi_flattened], dim=-1)  # (batch_size, feature_encoded_dim)
        value_input = torch.cat([features_encoded_major_flattened, latent_vf_flattened], dim=-1)   # (batch_size, feature_encoded_dim)

        actions_tensor = None
        if list_action is not None:
            actions_tensor = torch.tensor(list_action, dtype=torch.int32, device=self.device).unsqueeze(1)  # (batch_size, 1)

        actions, logprobs, entropies, values = self.ppo_model.policy_forward(policy_input, value_input, actions_tensor, deterministic=deterministic)

        return actions, logprobs, entropies, values
    

# TODO 由于可能有达到最大步会截断轨迹，因此会用到 bootstrap value if not done（那边是轨迹固定长度为 max_steps，可能有截断情况）。
    def forward_for_value(x): 
        pass
        


    def get_current_lstm_hidden_state(self):
        """算法实例从自身 LstmModel 类的成员中，获得当前的 LSTM 隐藏状态"""
        return self.lstm_model.get_current_hidden_state()
        
    # TODO GTL
    def compute_gae_and_rreturn(self, list_frame):
        """workflow 中收集完原始数据 Frames 到 collector 列表中后调用 agent.samples_process(list_frame=collector)"""
        # NOTE 唯一调用这个函数的地方是 feature 目录下 definition.py 里实现的 sample_process(list_frame) 函数，因此 frame 里必须携带
        pass

    

    def snapshot_model(self):
        """获取当前模型的快照"""

        # 调用 LstmPpoAlgorithm 类中的所有模型的 snapshot_model() 方法，并合并它们的状态字典。
        # feature_encoder = self.feature_encoder_model.snapshot_model()
        lstm = self.lstm_model.snapshot_model()
        ppo = self.ppo_model.snapshot_model()
        model_state_dict_cpu = {
            # "feature_encoder": feature_encoder,
            "lstm": lstm,
            "ppo": ppo,
        }
        return model_state_dict_cpu

    def load_model(self, model_state_dict_cpu):
        """初始化所有网络的参数"""
        # feature_encoder = model_state_dict_cpu["feature_encoder"]
        lstm = model_state_dict_cpu["lstm"]
        ppo = model_state_dict_cpu["ppo"]  

        # self.feature_encoder_model.load_model(feature_encoder)  
        self.lstm_model.load_model(lstm)
        self.ppo_model.load_model(ppo)
        pass 



# TODO GTL
    def learn(self, list_sample_data):
        """
        训练方法，输入是一个 SampleData 类型元素的列表，其中元素类型为 SampleData。每个 SampleData 对象对应一次转移过程，且包含更新网络所需的全部信息种类。

        该方法分为几个步骤：
        1. 展平数据
        2. 进入循环：for epoch, for mb
        shuffle 下标顺序
        new_infos <- self.forward_a_step()
        算 loss
        反向传播更新网络参数

        """
        
        batch_size = len(list_sample_data)
        assert(batch_size > 0), "list_sample_data must not be empty"
        # if batch_size == 0:
            # self.logger.warning("list_sample_data is empty, skipping learning step.")
            # return
        

        # 收集各个字段的数据
        b_obs = [ObsData(sample_data.feature, sample_data.lstm_state_pi, sample_data.lstm_state_vf) for sample_data in list_sample_data]
        b_log = [sample_data.logprob for sample_data in list_sample_data]
        b_entropy = [sample_data.entropy for sample_data in list_sample_data]
        b_advantage = [sample_data.advantage for sample_data in list_sample_data]
        b_rreturn = [sample_data.rreturn for sample_data in list_sample_data]
        b_value = [sample_data.value for sample_data in list_sample_data]
        b_action = [sample_data.action for sample_data in list_sample_data]
        b_done = [sample_data.done for sample_data in list_sample_data]
        
        # 转换为张量
        b_logprobs = torch.stack(b_log)
        b_entropies = torch.stack(b_entropy)
        b_advantages = torch.stack(b_advantage)
        b_returns = torch.stack(b_rreturn)
        b_values = torch.stack(b_value)
        b_actions = torch.stack(b_action)
        b_dones = torch.stack(b_done)
        
        # 标准化 advantages
        b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)
        
        clipfracs = []
        #clipfracs: 用于记录每个minibatch中被裁剪的比例
        for epoch in range(self.config.update_epoch):
            np.random.shuffle(b_inds)  # 随机打乱下标顺序
            for start in range(0, batch_size, self.config.minibatch_size):
                end = start + self.config.minibatch_si
                
                # TODO 计算三个 loss；读 agent_ppo 参考实现，搞清楚怎么反向传播
