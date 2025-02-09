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


class Trajectory:
    def __init__(self, env_id, label) -> None:
        self.env_id = env_id
        self.label = label
        self.infos = []
        self.states = []
        self.steps = []
        self.length = 0
        self.set_label()

    def set_label(self):

        if self.env_id == "Hopper-modified-Z-v0":
            self.x_label = "x(m)"
            self.y_label = "z(m)"
            self.title = "Hopper with Z-Preference"

        if self.env_id == "Swimmer-modified-Angle-v0":
            self.x_label = "step"
            self.y_label = "angle(rad)"
            self.title = "Swimmer with Angle-Constraint"

        if self.env_id == "Ant-modified-Y-v0":
            self.x_label = "x(m)"
            self.y_label = "y(m)"
            self.title = "Ant with Y-Preference"

        if self.env_id == "HalfCheetah-modified-Angle-v0":
            self.x_label = "step"
            self.y_label = "angle(rad)"
            self.title = "HalfCheetah with Angle-Constraint"




    def update(self, info, state) -> None:
        self.infos.append(info)
        self.states.append(state)
        self.steps.append(self.length)
        self.length += 1

    def special_draw(self, ax) -> None:
        x = []
        y = []

        if self.env_id == "Hopper-modified-Z-v0":
            for i in range(len(self.states) - 1):
                info = self.infos[i]
                state = self.states[i]
                step = self.steps[i]
                x.append(info["x_position"])
                y.append(state[0])
            ax.scatter([x[-1]], [y[-1]], color="red")


        if self.env_id == "Swimmer-modified-Angle-v0":
            for i in range(len(self.states) - 1):
                info = self.infos[i]
                state = self.states[i]
                step = self.steps[i]
                x.append(step)
                y.append(state[1])
            x = np.array(x)
            y = np.array(y)
            ax.scatter([x[-1]], [y[-1]], color="red")

        if self.env_id == "Ant-modified-Y-v0":
            for i in range(len(self.states) - 1):
                info = self.infos[i]
                state = self.states[i]
                step = self.steps[i]
                x.append(info["x_position"])
                y.append(-info["y_position"]) # -y axis
            ax.scatter([x[-1]], [y[-1]], color="red")

        if self.env_id == "HalfCheetah-modified-Angle-v0":
            for i in range(len(self.states) - 1):
                info = self.infos[i ]
                state = self.states[i]
                step = self.steps[i]
                x.append(step )
                y.append(state[7])



            # ax.scatter([x[-1]], [y[-1]], color="red")



    def extract_metrics(self):
        x = []
        y = []

        if self.env_id == "Hopper-modified-Z-v0":
            bias = 1
            for i in range(len(self.states) - bias):
                info = self.infos[i]
                state = self.states[i]
                step = self.steps[i]
                x.append(info["x_position"])
                y.append(state[0])

        if self.env_id == "Swimmer-modified-Angle-v0":
            for i in range(len(self.states) - 1):
                info = self.infos[i]
                state = self.states[i]
                step = self.steps[i]
                x.append(step)
                y.append(state[1])
            x = np.array(x)
            y = np.array(y)

        if self.env_id == "Ant-modified-Y-v0":
            bias = 1
            for i in range(len(self.states) - bias):
                info = self.infos[i]
                state = self.states[i]
                step = self.steps[i]
                x.append(info["x_position"])
                y.append(-info["y_position"]) # -y axis


        if self.env_id == "HalfCheetah-modified-Angle-v0":
            bias = 1
            for i in range(len(self.states) - bias):
                info = self.infos[i]
                state = self.states[i]
                step = self.steps[i]
                x.append(step)
                y.append(state[7])
            x = np.array(x)
            y = np.array(y)

        return np.array(x), np.array(y)


class Trajectories:
    def __init__(self) -> None:
        self.trajs = []

    def update(self, traj: Trajectory) -> None:
        self.trajs.append(traj)

    def visualize(self):
        font = {'family' : 'Times New Roman', 'weight' : 'normal', 'size'   : 15 }
        traj = self.trajs[0]
        fig, ax = plt.subplots()
        ax.set_xlabel(traj.x_label, font)
        ax.set_ylabel(traj.y_label, font)
        ax.set_title(traj.title, font)
        plt.grid()
        # ax.set_title(traj.env_id)
        

        for traj in self.trajs:
            x, y = traj.extract_metrics()
            ax.plot(x, y, label=traj.label)
            traj.special_draw(ax)
        
        ax.legend()
        # plt.savefig("./graphs/"+traj.env_id+".eps", dpi=1000)
        plt.savefig("./graphs/"+traj.env_id+".png", dpi=100)
        # plt.savefig("./graphs/"+traj.env_id+".svg", dpi=1000)











