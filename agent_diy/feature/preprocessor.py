from kaiwu_agent.utils.common_func import create_cls, attached
from agent_diy.conf.conf import Config

import numpy as np

ObsData = create_cls("ObsData", feature=None, legal_actions=None)
SampleData = create_cls("SampleData", rewards=None, dones=None)


def single_observation_process(self, obs, extra_info):
    game_info = extra_info["game_info"]
    # 合法动作
    local_view = game_info['local_view']
    legal_actions = []
    for i, view in enumerate([local_view[8], local_view[18], local_view[12], local_view[14]]): # UP DOWN LEFT RIGHT
        if view != 0:
            legal_actions.append(i + 1)
    
    pos = [game_info["pos_x"], game_info["pos_z"]]

    # 智能体当前位置相对于宝箱的距离(离散化)
    end_treasure_dists = obs["feature"]

    # Feature #5: Graph features generation (obstacle information, treasure information, endpoint information)
    # 图特征生成(障碍物信息, 宝箱信息, 终点信息)
    local_view = [game_info["local_view"][i : i + 5] for i in range(0, len(game_info["local_view"]), 5)]
    obstacle_map, treasure_map, end_map = [], [], []
    for sub_list in local_view:
        obstacle_map.append([1 if i == 0 else 0 for i in sub_list])
        treasure_map.append([1 if i == 4 else 0 for i in sub_list])
        end_map.append([1 if i == 3 else 0 for i in sub_list])

    # Feature #6: Conversion of graph features into vector features
    # 图特征转换为向量特征
    obstacle_flat, treasure_flat, end_flat = [], [], []
    for i in obstacle_map:
        obstacle_flat.extend(i)
    for i in treasure_map:
        treasure_flat.extend(i)
    for i in end_map:
        end_flat.extend(i)

    feature = np.concatenate(
        [
            pos,
            end_treasure_dists,
            obstacle_flat,
            treasure_flat,
            end_flat,
        ]
    )
    
    obs_data = ObsData(feature=feature, legal_actions=legal_actions)
    return obs_data

# SampleData <-> NumPy 转换规范
# 数据结构约定: 
# 我们将一组 SampleData 对象编码为一个 二维 NumPy 数组，其形状为: (N, M)
# N：并行环境数量(多个环境同时运行)
# M：每个 SampleData 实例中所有属性按固定顺序拼接后的总长度
# 间隔与截断规则:
# 不同属性用 -np.inf 填充
# 所有数值在编码前强制截断到有限范围(如 [-1e6, 1e6]), 防止出现 inf / nan。
# 例如:
# SampleData 有两属性 rewards([[1, 2, 3], [3, 2, 1]])；features([[1, 1, 1, 1, 1], [2, 2, 2, 2, 2]])
# 则 M = 8, N = 2
# 对应的 NumPy 数组为 [[1, 2, 3, -inf, 1, 1, 1, 1, 1], [3, 2, 1, -inf, 2, 2, 2, 2, 2]]

# 对于PPO算法，属性拼接顺序
# | 1  | feature          | feature_dim | 当前观测特征                       
# | 2  | next_feature     | feature_dim | 下一步观测特征                     
# | 3  | action           | action_dim  | 执行的动作（one-hot 或索引）       
# | 4  | log_prob         | action_dim  | 动作的对数概率                     
# | 5  | reward           | 1           | 即时奖励                          
# | 6  | done             | 1           | 是否结束（0 或 1）                   
# | 7  | value            | 1           | 状态价值估计           
# | 8  | advantage        | 1           | 广义优势估计（GAE）      
# | 9  | hidden_states_pi | lstm_hidden | Actor LSTM 隐藏状态     
# | 10 | cell_states_pi   | lstm_hidden | Actor LSTM 细胞状态    
# | 11 | hidden_states_vf | lstm_hidden | Critic LSTM 隐藏状态    
# | 12 | cell_states_vf   | lstm_hidden | Critic LSTM 细胞状态    

ATTR_TABLE = {
    'feature': Config.feature_dim, 
    'next_feature': Config.feature_dim, 
    'action': Config.action_dim, 
    'log_prob': Config.action_dim,
    'reward': 1, 
    'done': 1, 
    'value': 1, 
    'advantage': 1,
    'hidden_states_pi': (Config.N_LSTM_LAYERS, Config.N_ENVS, Config.LSTM_HIDDEN_SIZE), 
    'cell_states_pi': (Config.N_LSTM_LAYERS, Config.N_ENVS, Config.LSTM_HIDDEN_SIZE), 
    'hidden_states_vf': (Config.N_LSTM_LAYERS, Config.N_ENVS, Config.LSTM_HIDDEN_SIZE), 
    'cell_states_vf': (Config.N_LSTM_LAYERS, Config.N_ENVS, Config.LSTM_HIDDEN_SIZE)
}
N_ENVS = Config.N_ENVS  # 并行环境数量
SEP = -np.inf  # 属性间分隔符

@attached
def SampleData2NumpyData(sample: SampleData) -> np.ndarray:
    rows = []
    for env_id in range(N_ENVS):
        parts = []
        for name, shape in ATTR_TABLE.items():
            arr = getattr(sample, name)[env_id]  # (shape,)
            flat = np.asarray(arr, dtype=np.float32).ravel() # 展平为一维数组
            parts.append(flat)
            parts.append([SEP])
        flat_env = np.concatenate(parts[:-1])  # 去掉最后一个 SEP
        rows.append(flat_env)
    return np.stack(rows)  # (n_envs, M)

@attached
def NumpyData2SampleData(data: np.ndarray) -> SampleData:
    sample_dict = {}
    ptr = 0
    for name, shape in ATTR_TABLE.items():
        if isinstance(shape, int):
            dim = shape
        else:
            dim = np.prod(shape)
        # 切片 -> reshape
        val = data[:, ptr:ptr + dim]
        ptr += dim + 1  # +1 跳过 SEP
        val = val.reshape((N_ENVS,) + (shape if isinstance(shape, tuple) else (shape,)))
        sample_dict[name] = val
    return SampleData(**sample_dict)