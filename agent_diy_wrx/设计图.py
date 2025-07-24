# pylint: disable=all
# type: ignore
# mypy: ignore-errors

import torch
from torch import nn
import numpy as np

ObsData = create_cls(
    "ObsData",
    feature=None,       # Used to design this way: (seq_len=1, feature_dim, ). Now, simply (feature_dim, ) 
    lstm_state_pi=None, # [0]: h, [1]: c, what shape ??
    lstm_state_vf=None, # [0]: h, [1]: c, shape ??
)

ActData = create_cls(           # s -> s' by action a with reward r and d'
    "ActData",
    action=None,                # a
    logprob=None,               # pi(a|s)
    entropy=None,               # 这步选择做出来时，四个动作概率分布的熵，用于后续计算 entropy loss
    value=None,                 # s 时的 value，训练时要算新的（value loss 要新旧对比）。workflow 里就是把 act_data 里记录的 value 填到这步转移的数据帧里的。
)

SampleData = create_cls(
    "SampleData",               # s -> s' by action a with reward r and d'
    feature=None,               # s
    lstm_state_pi=None,         # s 时的 actor lstm 隐状态 (h,c)，训练时不翻新，off-policy
    lstm_state_vf=None,         # s 时的 critic lstm 隐状态 (h,c)，训练时不翻新，off-policy
    action=None,                # a
    logprob=None                # pi(a|s), 训练时要算新的（advantage clip 要求 ratio 要新旧对比）
    entropy=None,               # H(·|s)，训练时要算新的（entropy loss 要新旧对比）
    advantage=None,             # 轨迹收集完毕后立刻计算，训练时不翻新，off-policy
    rreturn=None,               # 跟随 GAE 计算的折扣回报，训练时不翻新，off-policy
    value=None,                 # 采取动作转移状态之前的状态，训练时要算新的（value loss 要新旧对比）
    done=None                   # d'
) 
# model.get_action_and_value() 时，要用到（不翻新的）feature，lstm_state_pi 和 lstm_state_vf，得到翻新的 values, logprobs, entropies。 



class Config:
    feature_dim = 88
    action_dim = 4

    feature_encoded_dim = 64    # 特征编码器输出的特征维度，后续 LSTM 网络的输入维度

    lstm_pi_seq_len = lstm_vf_seq_len = lstm_seq_len = 1

    lstm_pi_hidden_dim = lstm_vf_hidden_dim = 128
    lstm_pi_num_layers = lstm_vf_num_layers = 1

        
    # algo related

    gamma = 0.99

    update_epoch = 15 
    minibatch_size = 64
    clip_coef = 0.2
    ent_coef = 0.01          # 熵损失系数
    vf_coef = 0.5            # 值函数损失系数  
    pg_coef = 1.0            # 策略梯度系数,暂时没用到
    max_grad_norm = 0.5      # 梯度裁剪的最大范数

    # TODO add more













################################################################################################################

class Agent(BaseAgent):
    def __init__(self, agent_type="player", device=None, logger=None, monitor=None) -> None:
        self.device = device
        self.logger = logger
        self.monitor = monitor

        self.algo = LstmPpoAlgorithm(
            config=Config,
            device=self.device,
            logger=self.logger,
            monitor=self.monitor,
        )

    def observation_process(self, obs, extra_info) -> ObsData:
        # NOTE 更多的特征处理，可以在 preprossessor.py 里开发，然后在这里汇总成 ObsData 对象返回提交。
        feature = RawObservation2FeatureVector(obs, extra_info)
        current_lstm_state_pi, current_lstm_state_vf = self.algo.get_current_lstm_hidden_state()

        # 严格遵守 API：要返回 ObsData 类型对象（按照手册，要用 create_cls 函数定义）
        return ObsData(
            feature=feature, 
            lstm_state_pi=current_lstm_state_pi, 
            lstm_state_vf=current_lstm_state_vf
        )
    
    # TODO 读完手册后补充实现这个本来看似冗余的方法。游戏变复杂，要重新设计动作空间，或许真的需要解包 ActData 函数。
    # def action_process(self, act_data: ActData) -> int:
    #     return ActData.action
        

    @predict_wrapper    
    def predict(self, list_obs_data) -> [ActData]:
        """
        输入是一个 ObsData 类型元素的列表（之前原始观测已经被 observation_process 处理过；其实只有一个元素），输出是一个 ActData 类型元素的列表（其实只有一个）。
        
        该方法调用 agent.forward_a_step(..)，其中参数只有 ObsData 列表，得到动作相关信息的各自列表。
        为由于该函数在 rollout 阶段被调用，需要随机采样动作，故 actions = None, deterministic=False
        """
        actions, logprobs, entropies, values = self.algo.forward_a_step(
            list_obs_data=list_obs_data, 
            list_action=None,
            deterministic=False
        )
        
        return [
            ActData(
                action=action,
                logprob=logprob,
                entropy=entropy,
                value=value
            )
            for action, logprob, entropy, value in zip(actions, logprobs, entropies, values)
        ]



    @exploit_wrapper
    def exploit(self, observation) -> list[int]:
        """
        输入是一个原始数据（包括obs和extra_info），输出是一个动作列表（其实只含有一个动作）

        该方法调用 agent.forward_a_step(..)，其中参数为 ObsData 列表 + 确定性选择动作。返回值为动作相关信息的各自列表。
        为由于该函数在评估阶段被调用，需要贪心采样动作，故 actions = None, deterministic=True
        """
        obs, extra_info = observation["obs"], observation["extra_info"]
        ObsData = self.observation_process(obs, extra_info)
        actions, _, _, _ = self.algo.forward_a_step(list_obs_data=[ObsData], deterministic=True)
        action = actions[0].item()  # actions 是一个一维 tensor (batch_size=1, )，取第一个元素并转换为 int
        return action 

    @load_model_wrapper
    def load_model(self, id="latest"):
        # To load the model, it can consist of multiple files,
        # and it is important to ensure that each filename includes the "model.ckpt-id" field.
        # 加载模型, 可以是多个文件, 需要确保每个文件名里包括了model.ckpt-id字段
        model_file_path = f"{path}/model.ckpt-{str(id)}.pkl"

        # Load the model's state dictionary from the CPU
        # 从CPU加载模型的状态字典
        model_state_dict_cpu = torch.load(model_file_path, map_location=self.device)
        self.algo.load_model(model_state_dict_cpu)

    @save_model_wrapper
    def save_model(self, id="latest"):
        # To save the model, it can consist of multiple files,
        # and it is important to ensure that each filename includes the "model.ckpt-id" field.
        # 保存模型, 可以是多个文件, 需要确保每个文件名里包括了model.ckpt-id字段
        model_file_path = f"{path}/model.ckpt-{str(id)}.pkl"

        # Copy the model's state dictionary to the CPU
        # 将模型的状态字典拷贝到CPU
        model_state_dict_cpu = self.algo.snapshot_model()
        torch.save(model_state_dict_cpu, model_file_path)

    
    @learn_wrapper
    def learn(self, list_sample_data):
        # XXX 我猜 Learner 可能会实例化一个 Agent 类对象，并且自动调用 agent.load_model() 方法加载模型池里最新模型。但此猜测有一个缺陷：如果是这个机制，则无法加载指定模型，这可能是有人不希望的。所以，我还是手动加载一下模型。
        # self.load_model(id="latest")  # Load latest model
        self.algo.learn(list_sample_data)
        # self.save_model()



##################################################################################################################
# Algorithm/algorithm.py

from agent_diy.conf.conf import Config


class LstmPpoAlgorithm:
    # TODO GTL 增加更多算法相关的参数。注意别漏技巧，比如学习率退火。
    # 在这里设置的，都是类属性，无需实例化类对象，只需 from .. import LstmPpoAlgorithm 即可调用
    self.gamma = Config.gamma
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



#######################################################################################################################
# Model

def _layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class FeatureEncoderModel(nn.Module):
    def __init__(
            self, 
            device, 
            logger, monitor,
            input_dim,
            output_dim,
        ):
        super().__init__()
        self.device = device
        self.logger = logger
        self.monitor = monitor

        # Feature Encoder Network Parameters
        self.input_dim = input_dim
        self.output_dim = output_dim

        # FeatureEncoder network structure.
        # NOTE For now, actor and critic share the same feature encoder network.
        self.featureEncoder_network = nn.Sequential(
            _layer_init(nn.Linear(self.input_dim, 128)),
            nn.Tanh(), # or nn.ReLU(), which is better?
            nn.Linear(128, self.output_dim)
        )


    def forward(self, x):
        """
        Forward pass through the feature encoder network.

        x: (b, feature_dim, )  # 输入特征向量，形状为 (batch_size, feature_dim)
        返回值： (b, feature_encoded_dim, )  # 输出 encode 后的特征向量
        """
        return self.featureEncoder_network(x)
    
    def snapshot_model(self):
        """获取当前模型的快照"""
        # 返回当前 feature_encoder_network 的状态字典（网络参数）
        return self.featureEncoder_network.state_dict()
    
    def load_model(self, model_state_dict_cpu):
        """
        Load the model state dictionary from CPU.
        """
        # 解读出 feature_encoder_network 网络的参数并加载。
        self.featureEncoder_network.load_state_dict(model_state_dict_cpu)

class LstmModel(nn.Module):
    def __init__(
            self, 
            config, 
            device, 
            logger, 
            monitor, 

            pi_input_size,
            pi_hidden_size,
            pi_num_layers,
            vf_input_size,
            vf_hidden_size,
            vf_num_layers,
            
            seq_len,
        ):
        super().__init__()
        self.config = config
        self.device = device
        self.logger = logger
        self.monitor = monitor

        self.pi_input_size = pi_input_size
        self.pi_hidden_size = pi_hidden_size
        self.pi_num_layers = pi_num_layers
        self.vf_input_size = vf_input_size
        self.vf_hidden_size = vf_hidden_size
        self.vf_num_layers = vf_num_layers

        self.seq_len = seq_len

        

        """
        1. 根据传入参数，定义两个 LSTM 网络的结构

        rollout 阶段：
        - 正常の输入：   (seq_len=1,  batch_size=1, *_input_size)
        - 隐藏层输入：   (num_layers, batch_size=1, *_hidden_size)
        - 输出：        (seq_len=1,  batch_size=1, *_hidden_size)

        learn 阶段： batch_size = configure_app.toml 中的 buffer_size  
        """

        # 好在初始化网络和 batch_size 无关 :)
        self.lstm_pi_network = nn.LSTM(
            input_size=self.pi_input_size,
            hidden_size=self.pi_hidden_size,
            num_layers=self.pi_num_layers,
        )

        self.lstm_vf_network = nn.LSTM(
            input_size=self.vf_input_size,
            hidden_size=self.vf_hidden_size,
            num_layers=self.vf_num_layers,
        )

        """
        2. 初始化两个网络共4个隐藏层

        每种隐藏层的形状规定为 (num_layers, batch_size=1, *_hidden_size)。这样可以直接送入 LSTM 网络进行前向传播。

        NOTE 因为 LstmModel 对象记录的当前隐藏层，是服务于 rollout 阶段的，所以 batch_size=1。
        NOTE learn 阶段的 lstm 输入全部来自 ObsData，其过网所带来的隐藏层变化，已经不用记录了（off-policy）。此时批数据格式由 algo.forward_a_step() 保证对齐。
        """
        self.current_hidden_state_pi = [
            np.zeros((self.pi_num_layers, 1, self.pi_hidden_size), dtype=np.float32),
            np.zeros((self.pi_num_layers, 1, self.pi_hidden_size), dtype=np.float32)
        ]
        self.current_hidden_state_vf = [
            np.zeros((self.vf_num_layers, 1, self.vf_hidden_size), dtype=np.float32),
            np.zeros((self.vf_num_layers, 1, self.vf_hidden_size), dtype=np.float32)
        ]


    def pi_forward(self, x, hidden_state):
        """
        Forward pass through the LSTM networks.
        - x: 输入数据，形状为 (seq_len, batch_size, input_size)
        - hidden_state: 隐藏状态，形状为 (num_layers, batch_size, hidden_size)

        返回值：
        - output: LSTM 网络的输出，形状为 (seq_len, batch_size, hidden_size)
        - new_hidden_state: 更新后的隐藏状态，形状为 (num_layers, batch_size, hidden_size)
        """
        output, new_hidden_state = self.lstm_pi_network(x, hidden_state)
        return output, new_hidden_state

    def vf_forward(self, x, hidden_state):
        """
        Forward pass through the value function LSTM network.
        - x: 输入数据，形状为 (seq_len, batch_size, input_size)
        - hidden_state: 隐藏状态，形状为 (num_layers, batch_size, hidden_size)

        返回值：
        - output: LSTM 网络的输出，形状为 (seq_len, batch_size, hidden_size)
        - new_hidden_state: 更新后的隐藏状态，形状为 (num_layers, batch_size, hidden_size)
        """
        output, new_hidden_state = self.lstm_vf_network(x, hidden_state)
        return output, new_hidden_state

    def get_current_hidden_state(self):
        """
        返回值当前的隐藏态，形状：
        - current_hidden_state_pi: [h, c]，每个是 (num_layers, batch_size=1, pi_hidden_size)
        - current_hidden_state_vf: [h, c]，每个是 (num_layers, batch_size=1, vf_hidden_size)
        """
        return self.current_hidden_state_pi, self.current_hidden_state_vf


    def snapshot_model(self):
        """获取当前模型的快照"""
        # 返回当前两个 LSTM 网络的状态字典（网络参数）。
        # NOTE 当前隐藏层，不需要快照？毕竟之后不会再加载。
        return {
            "lstm_pi_network": self.lstm_pi_network.state_dict(),
            "lstm_vf_network": self.lstm_vf_network.state_dict(),
        }
    
    def load_model(self, model_state_dict_cpu):
        # 解读出 lstm_pi_network 和 lstm_vf_network 网络的参数并加载。
        self.lstm_pi_network.load_state_dict(model_state_dict_cpu["lstm_pi_network"])
        self.lstm_vf_network.load_state_dict(model_state_dict_cpu["lstm_vf_network"])
        
        # NOTE 该方法在每轮 rollout 开始之前被调用，所以加载完网络后，还要初始化 current_hidden_state_pi 和 current_hidden_state_vf 为 相应形状的 0。
        self.current_hidden_state_pi = [
            np.zeros((self.pi_num_layers, 1, self.pi_hidden_size), dtype=np.float32),
            np.zeros((self.pi_num_layers, 1, self.pi_hidden_size), dtype=np.float32)
        ]
        self.current_hidden_state_vf = [
            np.zeros((self.vf_num_layers, 1, self.vf_hidden_size), dtype=np.float32),
            np.zeros((self.vf_num_layers, 1, self.vf_hidden_size), dtype=np.float32)
        ]
        


class PpoModel(nn.Module):
    def __init__(
        self, 
        config, 
        device, 
        logger, 
        monitor,

        # network related parameters
        policy_net_input_dim,
        policy_net_output_dim,
        value_net_input_dim,    
    ):
        super().__init__()
        self.config = config
        self.device = device
        self.logger = logger
        self.monitor = monitor
        
        self.value_net_input_dim = value_net_input_dim
        self.policy_net_input_dim = policy_net_input_dim
        self.policy_net_output_dim = policy_net_output_dim
        
        # NOTE do not need algo-related parameters here (Algorithm class will handle them).
         

        # Build up the policy and value networks.
        self.value_network = nn.Sequential(
            _layer_init(nn.Linear(self.value_net_input_dim, 64)),
            nn.Tanh(),
            _layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            _layer_init(nn.Linear(64, 1), std=1.0),
        )

        self.policy_network = nn.Sequential(
            _layer_init(nn.Linear(self.policy_net_input_dim, 64)),
            nn.Tanh(),
            _layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            _layer_init(nn.Linear(64, self.policy_net_output_dim), std=0.01),
        )

    def policy_forward(self, pi_input_batch, vf_input_batch, actions=None, deterministic=False):
        """
        Forward pass (mainly) through the policy network.
        NOTE also involves the value network.

        - pi_input_batch: 输入数据，形状为 (batch_size, policy_input_dim)
        - vf_input_batch: 输入数据，形状为 (batch_size, value_input_dim)
        - actions: 可选参数，指定动作，形状为 (batch_size, )。如果为 None，则根据概率分布采样/取最大概率动作。
            - 若 deterministic=True，则取最大概率动作。
            - 若 deterministic=False，则从概率分布中采样动作。

        返回值：
        - output: 策略网络的输出，形状为 (batch_size, policy_output_dim)
        """
        logits = self.policy_network(pi_input_batch)
        probs = torch.distributions.categorical.Categorical(logits=logits)
        if actions == None:
            if deterministic:
                actions = probs.mode()
            else:
                actions = probs.sample()

        return actions, probs.log_prob(actions), probs.entropy(), self.value_network(vf_input_batch)

    def snapshot_model(self):
        """获取当前模型的快照"""
        # 返回当前 Policy 和 Value 网络的状态字典（网络参数）。
        return {
            "value_network": self.value_network.state_dict(),
            "policy_network": self.policy_network.state_dict(),
        }
    
    def load_model(self, model_state_dict_cpu):
        # 解读出 lstm_pi_network 和 lstm_vf_network 网络的参数并加载。
        self.value_network.load_state_dict(model_state_dict_cpu["value_network"])
        self.policy_network.load_state_dict(model_state_dict_cpu["policy_network"])

    



###########################################################################################
# feature/definition.py

# NOTE 这个函数必须实现在这里！如果关于算法的参数信息，请额外打包在frame 里。
# 同时注意 truncated 是可以表示达到最大帧数而游戏中断的。
# 因此计算 gae 的时候逃不掉 get_value 了，似乎需要在 Algorithm 类里额外实现 get_value 类似物。
# 需要读手册，修改 workflow，补充更多细节。
from agent_diy.algorithm.algorithm import LstmPpoAlgorithm

@attached
def samples_process(list_frame):
    """
    处理采样数据，将每一帧的特征转换为 SampleData 对象列表。
    """
    list_adv, list_rreturn = LstmPpoAlgorithm.compute_gae_and_rreturn(list_frame)
    
    # Note: training need advs and returns, not `reward`s.
    list_sample_data = [
        SampleData(
            feature=frame.obs_data.feature,
            lstm_state_pi=frame.obs_data.lstm_state_pi,
            lstm_state_vf=frame.obs_data.lstm_state_vf,
            action=frame.act_data.action,
            logprob=frame.act_data.logprob,
            entropy=frame.act_data.entropy,
            advantage=list_adv[i],
            rreturn=list_rreturn[i],
            value=frame.act_data.value,
            done=frame.done
        )
        for i, _ in enumerate(list_frame)
    ]

    return list_sample_data 

def reward_shaping(obs_data, _obs_data, extra_info, _extra_info, terminated, truncated, frame_no, score):

    # TODO GTL 抄其他简单的实现（xrq说不用搞得太复杂喧宾夺主，按照最原始最简单的来。但是我觉得也要考虑：稀疏环境的奖励，大部分时间都没有奖励函数，肯定不太好？先实现简单的再说）

    return reward=0




# My Helper Function
def RawObservation2FeatureVector(obs, extra_info): # -> (feature_dim, )
    
    # TODO GTL 把原始观测信息转为一位数组（就不用tensor化了，直接返回这个形状的数组吧）。我们再商量一下 tensor 转化的时机，尽量在比较深的抽象层统一转化，浅的就用 list 传递好了。

    return np.zeros((Config.feature_dim, ), dtype=np.float32)  









##################################################################################
# worlflow/workflow.py

# TODO 这俩函数作用是：分布式框架要求 agent.learn([g_data])发送数据之前，要先转 g_data 为 numpy.array 类型。
# 不急着看这个。这个是 ai 写的，后面我们检查一下。
@attached
def SampleData2NumpyData(g_data: SampleData):
    """将 SampleData 对象转换为 numpy 结构化数组"""
    import numpy as np
    
    # 创建结构化数组的数据类型定义
    dtype = [
        ('feature', 'f4', (g_data.feature.shape if g_data.feature is not None else (1,))),
        ('lstm_state_pi_h', 'f4', (g_data.lstm_state_pi[0].shape if g_data.lstm_state_pi else (1,))),
        ('lstm_state_pi_c', 'f4', (g_data.lstm_state_pi[1].shape if g_data.lstm_state_pi else (1,))),
        ('lstm_state_vf_h', 'f4', (g_data.lstm_state_vf[0].shape if g_data.lstm_state_vf else (1,))),
        ('lstm_state_vf_c', 'f4', (g_data.lstm_state_vf[1].shape if g_data.lstm_state_vf else (1,))),
        ('action', 'i4'),
        ('logprob', 'f4'),
        ('entropy', 'f4'),
        ('advantage', 'f4'),
        ('rreturn', 'f4'),
        ('value', 'f4'),
        ('done', 'bool'),
    ]
    
    # 创建结构化数组
    np_data = np.zeros(1, dtype=dtype)
    
    # 填充数据
    if g_data.feature is not None:
        np_data['feature'][0] = g_data.feature
    if g_data.lstm_state_pi is not None:
        np_data['lstm_state_pi_h'][0] = g_data.lstm_state_pi[0]
        np_data['lstm_state_pi_c'][0] = g_data.lstm_state_pi[1]
    if g_data.lstm_state_vf is not None:
        np_data['lstm_state_vf_h'][0] = g_data.lstm_state_vf[0]
        np_data['lstm_state_vf_c'][0] = g_data.lstm_state_vf[1]
    
    np_data['action'][0] = g_data.action if g_data.action is not None else 0
    np_data['logprob'][0] = g_data.logprob if g_data.logprob is not None else 0.0
    np_data['entropy'][0] = g_data.entropy if g_data.entropy is not None else 0.0
    np_data['advantage'][0] = g_data.advantage if g_data.advantage is not None else 0.0
    np_data['rreturn'][0] = g_data.rreturn if g_data.rreturn is not None else 0.0
    np_data['value'][0] = g_data.value if g_data.value is not None else 0.0
    np_data['done'][0] = g_data.done if g_data.done is not None else False
    
    return np_data

@attached
def NumpyData2SampleData(s_data: np.array):
    """将 numpy 结构化数组转换为 SampleData 对象"""
    
    # 重构 LSTM 状态元组
    lstm_state_pi = None
    if 'lstm_state_pi_h' in s_data.dtype.names and 'lstm_state_pi_c' in s_data.dtype.names:
        lstm_state_pi = (s_data['lstm_state_pi_h'][0], s_data['lstm_state_pi_c'][0])
    
    lstm_state_vf = None
    if 'lstm_state_vf_h' in s_data.dtype.names and 'lstm_state_vf_c' in s_data.dtype.names:
        lstm_state_vf = (s_data['lstm_state_vf_h'][0], s_data['lstm_state_vf_c'][0])
    
    return SampleData(
        feature=s_data['feature'][0] if 'feature' in s_data.dtype.names else None,
        lstm_state_pi=lstm_state_pi,
        lstm_state_vf=lstm_state_vf,
        action=s_data['action'][0] if 'action' in s_data.dtype.names else None,
        logprob=s_data['logprob'][0] if 'logprob' in s_data.dtype.names else None,
        entropy=s_data['entropy'][0] if 'entropy' in s_data.dtype.names else None,
        advantage=s_data['advantage'][0] if 'advantage' in s_data.dtype.names else None,
        rreturn=s_data['rreturn'][0] if 'rreturn' in s_data.dtype.names else None,
        value=s_data['value'][0] if 'value' in s_data.dtype.names else None,
        done=s_data['done'][0] if 'done' in s_data.dtype.names else None,
    )




# TODO 读新版手册，对照 agent_ppo 修改工作流细节
# 比如问到了“最大步数且没到终点，会让 truncated=1”，那么不仅 reward 的设计要狠狠惩罚，而且这条轨迹不能丢掉，需要 bootstrap if not done，也就是实现 get_value 类似物。
@attached
def workflow(envs, agents, logger=None, monitor=None):
    """
    Users can define their own training workflows here
    用户可以在此处自行定义训练工作流
    """

    try:
        # Read and validate configuration file
        # 配置文件读取和校验
        usr_conf = read_usr_conf("agent_diy/conf/train_env_conf.toml", logger)
        if usr_conf is None:
            logger.error("usr_conf is None, please check agent_diy/conf/train_env_conf.toml")
            return
        
        # Initializing monitoring data
        # 监控数据初始化
        monitor_data = {
            "reward": 0,
            "diy_1": 0,
            "diy_2": 0,
            "diy_3": 0,
            "diy_4": 0,
            "diy_5": 0,
        }
        last_report_monitor_time = time.time()

        logger.info("Start Training...")
        start_t = time.time()
        last_save_model_time = start_t


        # Training loop
        env = envs[0]
        agent : Agent = agents[0]

        num_epochs = 1000
        num_episodes_per_epoch = 1000

        # XXX 我觉得这里就要先保存一下模型？（即保存初始化的数据，防止 run_episodes 中没有模型可以加载）
        agent.save_model()  # Save initial model

        for epoch in range(num_epochs):
            for g_data in run_episode(num_episodes_per_epoch, env, agent, logger, monitor, usr_conf):
                agent.learn(g_data)
                g_data.clear()

            now = time.time()
            if now - last_save_model_time > 300:
                agent.save_model()
                last_save_model_time = now
            



    except Exception as e:
        raise RuntimeError(f"workflow error")


def run_episode(n_episodes, env, agent : Agent, loggger=None, monitor=None, usr_conf=None):
    for episode in range(n_episodes):
        done = False
        collector = list()
        
        obs, extra_info = env.reset(usr_conf=usr_conf)
        obs_data = agent.observation_process(obs, extra_info) # raw data -> ObsData
        agent.load_model(id="latest")

        while not done:
            """XXX I think obs_data should be updated here, not in the while loop XXX"""
            act_data = agent.predict(list_obs_data=[obs_data])[0] # list_obs_data and [0] is for KOG 3v3, here we only need 1 obs_data to predict 1 (the first) act_data 
            

            # action = agent.action_process(act_data)
            action = act_data.action

            frame_no, _obs, score, terminated, truncated, _extra_info = env.step(action)
            
            
            # Disaster recovery
            # Give up current collector and straight to next episode 
            if _obs == None:
                # raise RuntimeError("env.step return None obs")
                break
            if truncated and frame_no:
                break
            
            _obs_data = agent.observation_process(_obs, _extra_info)
            reward = reward_shaping(obs_data, _obs_data, extra_info, _extra_info, terminated, truncated, frame_no, score)
            done = terminated or truncated

            # Construct sample
            frame = Frame(
                obs_data=obs_data,
                _obs_data=_obs_data,
                act_data=act_data,
                reward=reward,
                done=done,
            )

            collector.append(frame)

            if done:
                if len(collector) > 0:
                    # XXX I think samples_process should be implemented in Agent
                    # collector = sample_process(collector)
                    collector = agent.samples_process(collector)
                    yield collector

                break

            obs_data = _obs_data
