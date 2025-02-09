import random
import numpy as np
import json
import os
import time

from tqdm import trange
from matplotlib import pyplot as plt

import mppi as mppi
from utils.load import load_dynamics, load_prior, load_para
from utils.dynamics import StatePredictor

from tqdm import trange
from matplotlib import pyplot as plt
from stable_baselines3.common.env_util import make_vec_env

from utils.visualizer import Trajectory, Trajectories
from utils.metric import Metric, Metrics, EvalLog

class Evaluator:
    def __init__(self) -> None:
        self.eval_log = EvalLog()
        self.planner = None
        self.env_id = None
        self.env = None

    def set_eval(self, env_id, network_path, hyperpara_arg, device="cuda") -> None:
        # Load hyperparameters
        self.network_parameters = load_para(network_path, id=env_id)
        self.hyperpara = hyperpara_arg

        # Set up environment
        self.env_id = env_id
        self.env = make_vec_env(env_id, n_envs=1, env_kwargs=self.network_parameters["env_kwargs"], wrapper_class=None)
        self.env.reset()

        # Load prior policy and dynamics
        prior_policy = load_prior(self.network_parameters["prior_policy_path"], device=device)
        dynamics = load_dynamics(self.network_parameters["dynamics_path"], env_id)
        self.planner = mppi.MPPI(env_id=env_id, env=self.env, dynamics=dynamics, prior_policy=prior_policy, planner_para=self.hyperpara)
        
    def eval(self, eval_arg, planning_arg, log_path=None, verbose=1, visualize=False) -> None:
        metric = Metric()
        self.eval_log.reset()
        # Episodes Loop
        nu = self.env.action_space.shape[0]
        for ep in trange(eval_arg["episodes"]):
            if eval_arg["seed"] is not None:
                self.env.seed(eval_arg["seed"])
                np.random.seed(eval_arg["seed"])
                random.seed(eval_arg["seed"])

            state = self.env.reset()
            metric.reset()
            if visualize:
                trajectory = Trajectory(env_id=self.env_id, label=planning_arg["type"])
            
            for it in trange(self.env.envs[0].max_ep_length):
                action = self.planner.command(state, planning_arg).detach().cpu().numpy().reshape(1, nu)

                state, reward, done, info = self.env.step(action)
                if visualize:
                    trajectory.update(info[0], state[0])

                # figure = self.env.render()
                metric.update(info)

                if done:
                    self.eval_log.total_metrics.update(metric, self.env_id)
                    if info[0]["success"]:
                        self.eval_log.success_metrics.update(metric, self.env_id)
                    if info[0]["fail"]:
                        self.eval_log.fail_metrics.update(metric, self.env_id)
                    if verbose > 0:
                        # print(f"Planning Frequency: {1/plan_time:.2f} Hz")
                        self.summary()
                    if visualize:
                        self.trajectories.update(trajectory)

                    break

        if log_path is not None:
            self.eval_log.save({"planning_arg": planning_arg, "network_parameters": self.network_parameters, "hyperpara": self.hyperpara}, log_path)


    def Eval(self, 
             eval_args, 
             hyperpara_args,
             planning_args, 
             env_ids, 

             hyperpara_path="parameters/hyperpara.yml", 
             network_path="parameters/networks.yml",
             planning_args_path="parameters/planning_args.yml", 
             eval_args_path="parameters/eval_args.yml", 
             device="cuda", verbose=1) -> None:


        # planning_args = [load_para(planning_args_path, id=planning_arg) for planning_arg in planning_args]
        # hyperpara_args = [load_para(hyperpara_path, id=hyperpara_arg) for hyperpara_arg in hyperpara_args]
        # eval_args = [load_para(eval_args_path, id=eval_arg) for eval_arg in eval_args]

        time_stamp = time.asctime()[4:19]
        basic_path = "eval_logs/" + time_stamp +"/"
        os.makedirs(basic_path, exist_ok=True)


        for env_id in env_ids:
            # Set up environment
            
            print(f"Start evaluation on {env_id}")
            for eval_arg_id in eval_args:
                eval_arg = load_para(eval_args_path, id=eval_arg_id)

                if eval_arg["visualize"]:
                    self.trajectories = Trajectories()
                print(f"Start evaluation with {eval_arg}")

                for planning_arg_id in planning_args:
                    planning_arg = load_para(planning_args_path, id=planning_arg_id)
                    print("Start evaluation with " + planning_arg["type"])

                    for hyperpara_arg_id in hyperpara_args:
                        hyperpara_arg = load_para(hyperpara_path, id=hyperpara_arg_id)
                        print(f"Start evaluation with {hyperpara_arg}")

                        env_path = basic_path + env_id + "/"
                        os.makedirs(env_path, exist_ok=True)
                        log_path = env_path + planning_arg["type"] + hyperpara_arg_id + "_" + str(eval_arg["episodes"]) + ".json"

                        self.set_eval(env_id, network_path, hyperpara_arg, device=device)
                        self.eval(eval_arg, planning_arg, log_path, verbose, visualize=eval_arg["visualize"])

                if eval_arg["visualize"]:
                    self.trajectories.visualize()



    def summary(self) -> None:
        self.eval_log.summary()
