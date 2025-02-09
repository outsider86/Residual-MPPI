import gym
import torch
import numpy as np


class SwimmerAngle(gym.Env):
    def __init__(self, **kwargs):
        super().__init__()
        self.inner_env = gym.make("Swimmer-v3", **kwargs)
        self.observation_space = self.inner_env.observation_space
        self.action_space = self.inner_env.action_space
        self.max_ep_length = 1000
        self.ep_step = 0

    def reset(self):
        self.ep_step = 0
        return self.inner_env.reset()

    # Penal large angle of the first angle
    def cost_R(self, state, action, full_mppi=False):
        if full_mppi:
            cost_R = (  1e-4*torch.square(action).sum(dim=1) # Action Cost
                        - state[:, 3]                        # Forward Reward
                        + torch.abs(state[:, 1])             # Add-on Cost on Angle
                         )
        else:
            cost_R = torch.abs(state[:, 1])
        return cost_R

    def addon_reward(self, observation):
        # encourage it towards -y axis
        addon_reward = -np.abs(observation[1])
        return addon_reward
    
    def addon_metric(self, observation, ):
        addon_metric = np.abs(observation[1])
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