import yaml

import torch

from stable_baselines3 import SAC
from utils.dynamics import StatePredictor

def load_para(para_path, id):
    with open(para_path) as file:
        content = file.read()
        data = yaml.load(content, Loader=yaml.FullLoader)
    return data[id]

def load_dynamics(dynamics_path, env_id):
    dynamics = torch.load(dynamics_path).eval()
    return dynamics

def load_prior(prior_path, env=None, device="cuda"):
    prior_policy = SAC.load(prior_path, env=env)
    prior_policy.actor.to(device)
    prior_policy.critic.to(device)
    prior_policy.actor.eval()
    prior_policy.critic.eval()

    return prior_policy