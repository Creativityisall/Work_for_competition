class Config:
    GAMMA = 0.99
    GAE_LAMBDA = 0.9
    EPSILON = 0.15
    LR_PPO = 5e-5
    T_MAX = 75
    LOSS_WEIGHT = {'policy': 1.0, 'value': 0.5, 'entropy': 0.001}

    FEATURE_DIM = 4
    ACTION_DIM = 2
    LSTM_HIDDEN_SIZE = 16
    N_LSTM_LAYERS = 1
    LATENT_DIM_PI = 32
    LATENT_DIM_VF = 32

    BUFFER_SIZE = 256
    N_ENVS = 1
    K_EPOCHS = 8
    UPDATE = 64
    MINIBATCH = 32