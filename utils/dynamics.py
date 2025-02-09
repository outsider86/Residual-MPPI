import torch
import torch.nn as nn
import numpy as np

class StatePredictor(nn.Module):
    def __init__(self, n_action, n_observation, hidden_layer):
        super(StatePredictor, self).__init__()
        self.n_observation = n_observation
        self.n_action = n_action
        self.model = nn.Sequential(
            torch.nn.Linear(n_action + n_observation, hidden_layer),
            torch.nn.Mish(),
            torch.nn.Linear(hidden_layer, hidden_layer),
            torch.nn.Mish(),
            torch.nn.Linear(hidden_layer, hidden_layer),
            torch.nn.Mish(),
            torch.nn.Linear(hidden_layer, hidden_layer),
            torch.nn.Mish(),
            torch.nn.Linear(hidden_layer, n_observation),
        )
    def forward(self, x):
        return self.model(x)

    # Calculate the next state in a residual manner (i.e. next_state = state + forward(state, action))
    def rollout(self, states, actions, env_id):
        # They are trained on a reduced state space (27dims) in 111 and the rest of the state space (84dim) are all zeros
        if env_id  == "Ant-modified-Y-v0":
            next_states = states[:, :27] + self.forward(torch.cat((states[:, :27], actions), dim=1))
            next_states = torch.cat((next_states, torch.zeros(next_states.shape[0], 84).to(device=states.device)), dim=1)
        else:
            next_states = states + self.forward(torch.cat((states, actions), dim=1))
        return next_states
