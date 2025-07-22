import torch
import numpy as np
from agent_diy.conf.conf import Config

# My Helper Function
def RawObservation2FeatureVector(obs, extra_info): # -> (feature_dim, )
    
    # TODO GTL 把原始观测信息转为一位数组（就不用tensor化了，直接返回这个形状的数组吧）。我们再商量一下 tensor 转化的时机，尽量在比较深的抽象层统一转化，浅的就用 list 传递好了。

    return np.zeros((Config.feature_dim, ), dtype=np.float32)  
