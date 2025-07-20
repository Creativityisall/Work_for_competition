import gymnasium as gym

class Env:
    def __init__(self):
        self.env = gym.make("CartPole-v1")
        self.current_frame_no = 0

    def reset(self):
        observation, info = self.env.reset()
        self.current_frame_no = 0
        extra_info = {"result_code": 0, "score_info": {"score": 0}}
        return observation, extra_info

    def step(self, action):
        _obs, reward, terminated, truncated, info = self.env.step(action)
        self.current_frame_no += 1
        _extra_info = {"result_code": 0, "score_info": {"score": reward}}
        return self.current_frame_no, _obs, terminated, truncated, _extra_info

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()