import torch
import CustomEnv

from utils.dynamics import StatePredictor
from utils.evaluator import Evaluator

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

dtype = torch.float32
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu") 
Exp = Evaluator()

# Select the test policy
planning_args = [
    "prior",
    "greedy_mppi",
    "full_mppi",
    "guided_mppi",
    "valued_mppi",
    "residual_mppi",
]

# Select the hyperparameters setting
hyperpara_args = [
    "Swimmer-original",
    # "Hopper-original",
    # "HalfCheetah-original",
    # "Ant-original",
]

# Select the test environment
env_ids = [
    "Swimmer-modified-Angle-v0", 
    # "Hopper-modified-Z-v0", 
    # "HalfCheetah-modified-Angle-v0",
    # "Ant-modified-Y-v0", 
    ]

# Select the evaluation mode
eval_args = [
    "demo",           # Run 1 episode with each policy
    # "test",           # Run 10  episodes with each policy
    # "ablation",       # Run 100  episodes with each policy
    # "visualize",        # Run 1 episode with each policy and visualize (you can only compare prior, guided_mppi, and residual_mppi for more clear visualization)
    # "fullexp",        # Run 500 episodes with each policy
]


# The experiment results would be saved in the eval_logs/
# The visualization output would be saved in the graphs/
Exp.Eval(eval_args=eval_args, hyperpara_args=hyperpara_args, planning_args=planning_args, env_ids=env_ids, device=device)