import gym
import torch
import numpy as np

class HopperZ(gym.Env):
    def __init__(self, **kwargs):
        super().__init__()
        self.inner_env = gym.make("Hopper-v3", **kwargs)
        self.observation_space = self.inner_env.observation_space
        self.action_space = self.inner_env.action_space
        self.max_ep_length = 1000
        self.ep_step = 0

        self.healthy_height = 0.7
        self.healthy_angle = 0.2
        self.healthy_state = 100

        self.basic_height = 1

    def reset(self):
        self.ep_step = 0
        return self.inner_env.reset()
    
    def cost_R(self, state, action, full_mppi):
        if full_mppi:
            max_state = torch.max(state[:, 1:], dim=1)[0]
            min_state = torch.min(state[:, 1:], dim=1)[0]
            alive = ((state[:, 0] > self.healthy_height).float() 
                    * (state[:, 1] > -self.healthy_angle).float() 
                    * (state[:, 1] < self.healthy_angle).float()
                    * (min_state > -self.healthy_state).float()
                    * (max_state < self.healthy_state).float()
                    )
            cost_R = (- alive                                  # Alive Reward
                      + 1e-3*torch.square(action).sum(dim=1)   # Action Cost
                      - state[:, 5]                            # Forward Reward
                      - (state[:, 0] - self.basic_height) * 10 # Add-on Reward on Height
                      )
        else:
            cost_R = - (state[:, 0] - self.basic_height) * 10
        return cost_R

    def addon_reward(self, observation):
        addon_reward = (observation[0] - self.basic_height) * 10
        return addon_reward
    
    def addon_metric(self, observation):
        addon_metric = observation[0]
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