from utils import create_cls

SampleData = create_cls("SampleData", rewards=None, dones=None)

def sample_process(sample):
    return SampleData(rewards=sample.rewards, dones=sample.dones)

def single_reward_shaping(frame_no, terminated, truncated, obs, next_obs, extra_info, next_extra_info, step):
    reward = 0
    reward = next_extra_info["score_info"]["score"]
    return reward

def reward_shaping(
    list_frame_no, 
    list_terminated, 
    list_truncated, 
    list_obs, 
    list_next_obs, 
    list_extra_info, 
    list_next_extra_info, 
    step
    ) -> list[int]:
    rewards = []
    for idx in range(len(list_frame_no)):
        reward = single_reward_shaping(
            list_frame_no[idx], 
            list_terminated[idx], 
            list_truncated[idx], 
            list_obs[idx], 
            list_next_obs[idx], 
            list_extra_info[idx], 
            list_next_extra_info[idx], 
            step
        )
        rewards.append(reward)

    return rewards