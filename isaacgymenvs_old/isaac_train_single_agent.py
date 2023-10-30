import sys
sys.path.insert(0, './..')
import torch.nn.functional as F
from torch import jit, nn
import isaacgym
import isaacgymenvs
import torch
import time
import numpy as np
from copy import deepcopy

# num_envs = 10


import torch.nn as nn
@torch.jit.script
def normalize(a, maximum, minimum):
    temp_a = 1.0/(maximum - minimum)
    temp_b = minimum/(minimum - maximum)
    temp_a = torch.ones_like(a)*temp_a
    temp_b = torch.ones_like(a)*temp_b
    return temp_a*a + temp_b

@torch.jit.script
def unnormalize(a, maximum, minimum):
    temp_a = maximum - minimum
    temp_b = minimum
    temp_a = torch.ones_like(a)*temp_a
    temp_b = torch.ones_like(a)*temp_b
    return temp_a*a + temp_b
class PolicyValueNetwork(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super(PolicyValueNetwork, self).__init__()
        self.fc1 = nn.Linear(obs_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.mean_head = nn.Linear(64, act_dim)
        self.log_std_head = nn.Linear(64, act_dim)
        self.value_head = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.nn.functional.elu(self.fc1(x))
        x = torch.nn.functional.elu(self.fc2(x))
        x = torch.nn.functional.elu(self.fc3(x))
        mean = self.mean_head(x)
        log_std = self.log_std_head(x)
        values = self.value_head(x)
        return mean, log_std, values

LOG_STD_MAX = 2
LOG_STD_MIN = -4
EPS = 1e-8

def initWeights(m):
    if isinstance(m, torch.nn.Linear):
        m.weight.data.normal_(0, 0.01)
        m.bias.data.normal_(0, 0.01)

class Policy(nn.Module):
    def __init__(self, args):
        super(Policy, self).__init__()

        self.state_dim = args['state_dim']
        self.action_dim = args['action_dim']
        self.hidden1_units = args['hidden_dims'][0]
        self.hidden2_units = args['hidden_dims'][1]
        self.activation = args['activation'].lower()
        self.log_std_init = args['log_std_init']

        self.fc1 = nn.Linear(self.state_dim, self.hidden1_units)
        self.fc2 = nn.Linear(self.hidden1_units, self.hidden2_units)
        self.act_fn = eval(f'F.{self.activation}')
        self.output_act_fn = torch.sigmoid

        self.fc_mean = nn.Linear(self.hidden2_units, self.action_dim)
        self.log_std = nn.Parameter(self.log_std_init*torch.ones(self.action_dim))


    def forward(self, x):
        x = self.act_fn(self.fc1(x))
        x = self.act_fn(self.fc2(x))
        mean = self.output_act_fn(self.fc_mean(x))
        log_std = self.log_std
        log_std = torch.clamp(log_std, min=LOG_STD_MIN, max=LOG_STD_MAX)
        std = torch.exp(log_std)
        return mean, log_std, std

    def initialize(self):
        for m_idx, module in enumerate(self.children()):
            module.apply(initWeights)


class Value(nn.Module):
    def __init__(self, args):
        super(Value, self).__init__()

        self.state_dim = args['state_dim']
        self.action_dim = args['action_dim']
        self.hidden1_units = args['hidden_dims'][0]
        self.hidden2_units = args['hidden_dims'][1]
        self.activation = args['activation'].lower()

        self.fc1 = nn.Linear(self.state_dim, self.hidden1_units)
        self.fc2 = nn.Linear(self.hidden1_units, self.hidden2_units)
        self.fc3 = nn.Linear(self.hidden2_units, 1)
        self.act_fn = eval(f'F.{self.activation}')


    def forward(self, x):
        x = self.act_fn(self.fc1(x))
        x = self.act_fn(self.fc2(x))
        x = self.fc3(x)
        x = torch.reshape(x, (-1,))
        return x

    def initialize(self):
        self.apply(initWeights)


class BasicAgent:
    def __init__(self, obs_dim, act_dim, action_bound_min, action_bound_max, device):
        self.policy = Policy({
            'state_dim': obs_dim,
            'action_dim': act_dim,
            'hidden_dims': [256, 128],
            'activation': 'elu',
            'log_std_init': 0.0  # Some initial value for log_std
        }).to(device)
        
        self.value = Value({
            'state_dim': obs_dim,
            'action_dim': act_dim,
            'hidden_dims': [256, 128],
            'activation': 'elu'
        }).to(device)
        
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=0.001)
        self.value_optimizer = torch.optim.Adam(self.value.parameters(), lr=0.0003)
        
        self.gamma = 0.99  # discount factor
        self.device = device
        self.action_bound_min = torch.tensor(action_bound_min, device=self.device)
        self.action_bound_max = torch.tensor(action_bound_max, device=self.device)
        self.max_kl = 1e-2
        self.policy_epochs = 1
        self.value_epochs = 1
        self.ent_coeff = 0.00
    def get_action(self, obs, is_train=True):
        with torch.no_grad():
            # mean, log_std, _ = self.model(obs)
            mean, log_std, _ = self.policy(obs)
            std = log_std.exp()
            normal = torch.distributions.Normal(mean, std)
            action = normal.sample()

            if is_train:
                noise = torch.randn(*mean.size(), device=self.device)
                action = self.unnormalizeAction(mean + noise*std)
            else:
                action = self.unnormalizeAction(mean)
        # return action
        return action, normal.log_prob(action).sum(dim=-1, keepdim=True)
    

    def normalizeAction(self, a:torch.Tensor):
        # print(f"normalizeAction: a.shape: {a.shape}")
        return normalize(a, self.action_bound_max, self.action_bound_min)
    def unnormalizeAction(self, a:torch.Tensor):
        # print(f"unnormalizeAction: a.shape: {a.shape}")
        return unnormalize(a, self.action_bound_max, self.action_bound_min)

    def train(self, obs, action, reward, next_obs, done, log_prob):
        # Get old policy outputs
        old_mean, old_log_std, _ = self.policy(obs)
        old_std = old_log_std.exp()
        
        # Get value outputs
        values = self.value(obs)
        next_values = self.value(next_obs)
        
        # Calculate the expected value
        reward = reward.view(-1, 1)
        done = done.view(-1, 1)
        values = values.view(-1, 1)
        next_values = next_values.view(-1, 1)
        expected_values = reward + self.gamma * next_values * (1 - done.float())

        # Get policy outputs
        mean, log_std, _ = self.policy(obs)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        
        # Calculate the policy loss using the advantage
        advantage = expected_values - values
        policy_loss = -(log_prob * advantage).mean()

        # Backpropagation for policy
        self.policy_optimizer.zero_grad()
        policy_loss.backward(retain_graph=True)  # No need to retain the graph
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)  # Gradient clipping
        self.policy_optimizer.step()

        # Recalculate the value loss
        value_loss = torch.nn.functional.mse_loss(values, expected_values.detach())
        
        # Backpropagation for value
        self.value_optimizer.zero_grad()
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.value.parameters(), max_norm=1.0)  # Gradient clipping
        self.value_optimizer.step()

        return policy_loss.item(), value_loss.item()
    def getGaesTargets(self, rewards, values, dones, fails, next_values):
        rewards = rewards.reshape((-1, self.num_envs))
        values = values.reshape((-1, self.num_envs))
        dones = dones.reshape((-1, self.num_envs))
        fails = fails.reshape((-1, self.num_envs))
        next_values = next_values.reshape((-1, self.num_envs))
        deltas = rewards + (1.0 - fails) * self.discount_factor * next_values - values
        gaes = deepcopy(deltas)
        for t in reversed(range(len(gaes))):
            if t < len(gaes) - 1:
                gaes[t] = gaes[t] + (1.0 - dones[t]) * self.discount_factor * self.gae_coeff * gaes[t + 1]
        targets = values + gaes
        return gaes.reshape(-1), targets.reshape(-1)
    # def train(self, obs, action, reward, next_obs, done, log_prob):
    #     # Convert tensors to numpy arrays for certain calculations
    #     rewards_np = reward.detach().cpu().numpy()
    #     dones_np = done.detach().cpu().numpy()

    #     # Normalize actions
    #     norm_actions_tensor = self.normalizeAction(action)

    #     # Get GAEs and Targets
    #     values_tensor = self.value(obs)
    #     next_values_tensor = self.value(next_obs)
    #     values_np = values_tensor.detach().cpu().numpy()
    #     next_values_np = next_values_tensor.detach().cpu().numpy()
    #     gaes, targets = self.getGaesTargets(rewards_np, values_np, dones_np, next_values_np)
    #     gaes_tensor = torch.tensor(gaes, device=self.device, dtype=torch.float).view(-1, 1)
    #     targets_tensor = torch.tensor(targets, device=self.device, dtype=torch.float).view(-1, 1)

    #     # Policy update
    #     old_means, old_log_stds, old_stds = self.policy(obs)
    #     old_dist = torch.distributions.Normal(old_means, old_stds)
    #     old_log_probs = torch.sum(old_dist.log_prob(norm_actions_tensor), dim=1)
        
    #     means, log_stds, stds = self.policy(obs)
    #     dist = torch.distributions.Normal(means, stds)
    #     log_probs = torch.sum(dist.log_prob(norm_actions_tensor), dim=1)
    #     ratios = torch.exp(log_probs - old_log_probs)
    #     clipped_ratios = torch.clamp(ratios, 1.0 - self.clip_value, 1.0 + self.clip_value)
    #     policy_loss = -(torch.min(gaes_tensor * ratios, gaes_tensor * clipped_ratios)).mean()
        
    #     self.policy_optimizer.zero_grad()
    #     policy_loss.backward(retain_graph=True)
    #     torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
    #     self.policy_optimizer.step()

    #     # Value update
    #     value_loss = (self.value(obs) - targets_tensor).pow(2).mean()
    #     self.value_optimizer.zero_grad()
    #     value_loss.backward()
    #     torch.nn.utils.clip_grad_norm_(self.value.parameters(), self.max_grad_norm)
    #     self.value_optimizer.step()

    #     return policy_loss.item(), value_loss.item()

    def update_from_rollout(self, rollout_data):
        observations = torch.stack(rollout_data['observations'])
        actions = torch.stack(rollout_data['actions'])
        rewards = torch.stack(rollout_data['rewards'])
        log_probs = torch.stack(rollout_data['log_probs'])

        # Calculate returns using rewards
        returns = []
        R = 0
        for r in reversed(rewards):
            R = r + self.gamma * R
            returns.insert(0, R)
        returns = torch.stack(returns)

        # Get policy outputs
        mean, log_std, _ = self.policy(observations)
        std = log_std.exp()
        policy_dist = torch.distributions.Normal(mean, std)

        # Get value outputs
        values = self.value(observations).view(-1)

        if isinstance(policy_dist, torch.distributions.Categorical):
            action_probs = policy_dist.probs.gather(1, actions.unsqueeze(-1))
        else:
            action_probs = policy_dist.log_prob(actions)
        ratio = (action_probs / log_probs).squeeze()
        ratio = ratio.mean(dim=-1)

        advantages = returns - values.squeeze()

        surrogate_loss = ratio * advantages
        value_loss = 0.5 * advantages.pow(2).mean()
        policy_loss = -surrogate_loss.mean()

        # Backpropagation for policy
        self.policy_optimizer.zero_grad()
        policy_loss.backward(retain_graph=True)  # Retain graph for value backpropagation
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)  # Gradient clipping
        self.policy_optimizer.step()

        # Backpropagation for value
        self.value_optimizer.zero_grad()
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.value.parameters(), max_norm=1.0)  # Gradient clipping
        self.value_optimizer.step()
