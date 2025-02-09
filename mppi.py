# This implementation is heavily inspired by the mppi-pytorch implementation of the UM-ARM Lab
# https://github.com/UM-ARM-Lab/pytorch_mppi

import math
import torch
from torch.distributions.multivariate_normal import MultivariateNormal

class MPPI():
    def __init__(self, 
                 env_id, env,
                 dynamics, prior_policy, 
                 planner_para,
                 device="cuda",
                 dtype=torch.float32,
                 ):
        # Basic Para
        self.device = device
        self.dtype = dtype

        # Env Para
        self.env = env
        self.env_id = env_id

        self.nx = env.observation_space.shape[0]
        self.nu = env.action_space.shape[0]
        self.u_min=torch.tensor(env.action_space.low, dtype=dtype, device=device)
        self.u_max=torch.tensor(env.action_space.high, dtype=dtype, device=device)

        # Planning Para
        self.planner_para = planner_para
        self.K = planner_para["num_samples"]
        self.T = planner_para["horizon"]
        self.gamma = planner_para["gamma"]
        self.lambda_ = planner_para["lambda_"]
        self.init_cov_diag = planner_para["init_cov_diag"]
        self.mppi_scale = planner_para["mppi_scale"]
        
        # Diagonal covariance matrix
        noise_sigma = self.init_cov_diag * torch.tensor(torch.diag(torch.ones(self.nu)), device=device, dtype=dtype)
        self.noise_sigma_inv = torch.inverse(noise_sigma)
        self.noise_dist = MultivariateNormal(torch.zeros(self.nu, dtype=dtype, device=device), covariance_matrix=noise_sigma)

        # Prior Policy and Dynamics
        self.prior_policy = prior_policy
        self.dynamics = dynamics.to(device)

    # logpi: prior log probability of the action given the state
    def _logpi(self, states, actions, full_mppi, without_log):
        if full_mppi or without_log:
            return 0
        else:
            # Built-in methods of StableBaseline3 implementation
            mean_actions, log_std, kwargs = self.prior_policy.actor.get_action_dist_params(states)
            self.prior_policy.actor.action_dist.actions_from_params(mean_actions, log_std)
            log_prob = self.prior_policy.actor.action_dist.log_prob(actions)
            return log_prob


    def _cost_R(self, state, action, full_mppi):
        return self.env.envs[0].cost_R(state, action, full_mppi).reshape(-1)

    # Implement it from the cost perspective, so it would be a negative sign comparing to the paper
    def command(self, state, planning_arg):
        with torch.no_grad():

            # Prior Policy
            if planning_arg["prior"]:
                ref_action, _ = self.prior_policy.predict(state, deterministic=True)
                return torch.tensor(ref_action, dtype=self.dtype, device=self.device)
            
            # Prior Policy Guided
            # Initialize the reference action sequence
            state = torch.tensor(state, dtype=self.dtype, device=self.device).reshape(1, self.nx)
            U = self.noise_dist.sample((self.T,))
            if planning_arg["prior_guided"]:
                actor_state = state.clone().reshape(1, -1)
                noise = self.noise_dist.rsample((self.K, self.T))
                for t in range(self.T):
                    actor_action = self.prior_policy.actor(actor_state, deterministic = True)
                    U[t] = torch.tensor(actor_action, dtype=self.dtype, device=self.device).reshape(1, -1)
                    actor_state = self.dynamics.rollout(actor_state, U[t].reshape(1, -1), self.env_id)
            else:
                # Full-MPPI: Use Zero-input as the reference action distribution
                U = torch.zeros_like(self.noise_dist.sample((self.T,)))

                # Scale the noise to make it explore more spaces rather than be confined within a really small range
                noise = self.noise_dist.rsample((self.K, self.T)) / self.init_cov_diag

            # Eval the prior nominal action sequence by initializing the first noise as zero
            if planning_arg["eval_prior"]:
                noise[0] = torch.zeros_like(noise[0], dtype=self.dtype, device=self.device)   

            # Get the perturbed action sequence distribution for rollout and evaluation
            perturbed_action = torch.clamp(U + noise, min=self.u_min, max=self.u_max)
            noise = perturbed_action - U
            


            # Rollout the action sequences to trajectories
            # Calculate the first two terms in the algorithm table
            rollout_states = state.repeat(self.K, 1)
            rollout_cost = torch.zeros(self.K, device=self.device, dtype=self.dtype)
            for t in range(self.T):
                u = perturbed_action[:, t]
                # Logpi
                logpi = self._logpi(rollout_states, u, planning_arg["full_mppi"], planning_arg["without_log"]) * self.mppi_scale
                rollout_states = self.dynamics.rollout(rollout_states, u, self.env_id)
                # r_R
                customization_cost = self._cost_R(rollout_states, u, planning_arg["full_mppi"])
                # Terminal Q
                if planning_arg["value"]:
                    terminal_q = self.prior_policy.critic(rollout_states, u)[0].reshape(-1) if t == self.T - 1 else 0
                else:
                    terminal_q = 0

                rollout_cost += (customization_cost - logpi - terminal_q) * math.pow(self.gamma, t)

            # This implementation detail comes from the mppi implementation of the UM-ARM https://github.com/UM-ARM-Lab/pytorch_mppi
            # Linked to src/mppi.py line 254
            # https://github.com/UM-ARM-Lab/pytorch_mppi/blob/a272c5174d690fbb6f890bce9dbdb695248643fe/src/pytorch_mppi/mppi.py#L254C71-L254C86

            # NOTE: The original mppi paper does self.lambda_ * self.noise @ self.noise_sigma_inv, but this biases the actions with low noise 
            # if all states have the same cost. With abs(noise) we prefer actions close to the nomial trajectory.
            if planning_arg["noise_abs_cost"]:
                action_cost = self.lambda_ * torch.abs(noise) @ self.noise_sigma_inv
            else:
                action_cost = self.lambda_ * noise @ self.noise_sigma_inv              # Like original mppi paper

             # Calculate the third term together of the action sequence cost
            perturbation_cost = torch.sum(U * action_cost, dim=(1, 2))
            cost_total = rollout_cost + perturbation_cost

            # Update the Action
            cost_total_non_zero = torch.exp( -(cost_total - torch.min(cost_total)) / self.lambda_ )  # constant would not effect the weight in theory, just to make it numerically stable
            omega = cost_total_non_zero / torch.sum(cost_total_non_zero)
            perturbations = []
            for t in range(self.T):
                perturbations.append(torch.sum(omega.view(-1, 1) * noise[:, t], dim=0))
            perturbations = torch.stack(perturbations)

            # Execute the Action
            U = torch.clamp(U + perturbations, min=self.u_min, max=self.u_max)
            return U[0]



