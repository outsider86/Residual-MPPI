import gym
import torch
import numpy as np

from gym.spaces import Box

class AntY(gym.Env):
    def __init__(self, **kwargs):
        super().__init__()
        self.inner_env = gym.make("Ant-v3", **kwargs)
        self.observation_space = self.inner_env.observation_space
        self.action_space = self.inner_env.action_space
        self.max_ep_length = 1000
        self.ep_step = 0

        self.safe_height = 0.2

    def reset(self):
        self.ep_step = 0
        observation = self.inner_env.reset()
        return observation
    
    # Encourage the -y direction, so there is a negetive sign here
    # The code in evaluation visualization is also designed on this setting
    def cost_R(self, state, action, full_mppi=False):
        if full_mppi:
            alive = (state[:, 0] > self.safe_height).float()
            cost_R = (- alive                                # Alive Reward
                      + 0.5*torch.square(action).sum(dim=1)  # Action Cost
                      - state[:, 13]                         # Forward Reward
                      + state[:, 14]                         # Add-on Reward on -v_y
                      )
        else:
            cost_R = state[:, 14]

        return cost_R


    def addon_reward(self, observation):
        addon_reward = -observation[14]
        return addon_reward
    
    def addon_metric(self, observation):
        addon_metric = -observation[14]
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



