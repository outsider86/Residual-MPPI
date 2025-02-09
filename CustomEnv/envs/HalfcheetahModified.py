import gym
import torch
import numpy as np


class CrippleCheetah(gym.Env):
    def __init__(self, **kwargs):
        super().__init__()
        self.inner_env = gym.make("HalfCheetah-v3", **kwargs)
        self.observation_space = self.inner_env.observation_space
        self.action_space = self.inner_env.action_space
        self.max_ep_length = 1000
        self.ep_step = 0

    def reset(self):
        self.ep_step = 0
        return self.inner_env.reset()
    
    def cost_R(self, state, action, full_mppi=False):
        if full_mppi:
            cost_R = (0.1*torch.square(action).sum(dim=1)  # Action Cost
                      - state[:, 8]                        # Forward Reward
                      + 10 * torch.abs(state[:, 7])        # Add-on Cost on Angle
                      )
        else:
            cost_R = torch.abs(state[:, 7]) * 10

        return cost_R

    def addon_reward(self, observation):
        addon_reward = - 10 * np.abs(observation[7])
        return addon_reward
    
    def addon_metric(self, observation):
        addon_metric = np.abs(observation[7])
        return addon_metric


    def step(self, action):
        observation, reward, terminated, info = self.inner_env.step(action)
        info["success"] = False
        info["fail"] = False
        self.ep_step += 1

        info["basic_reward"] = reward
        info["addon_reward"] = self.addon_reward(observation)
        info["total_reward"] = reward + info["addon_reward"]
        info["addon_metric"] = self.addon_metric(observation)

        if terminated:
            if self.ep_step == self.max_ep_length:
                info["success"] = True
            else:
                info["fail"] = True

        return observation, reward, terminated, info

    def render(self, mode, **kwargs):
        return self.inner_env.render(mode, **kwargs)

    def close(self):
        return self.inner_env.close()