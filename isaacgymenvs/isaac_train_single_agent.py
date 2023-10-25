import sys
sys.path.insert(0, './..')

import isaacgym
import isaacgymenvs
import torch
import time
import numpy as np

num_envs = 10

print("Creating environment 1...")
envs1 = isaacgymenvs.make(
    seed=0, 
    task="Jackal", 
    num_envs=num_envs, 
    sim_device="cuda:0",
    rl_device="cuda:0",
    graphics_device_id=0,
    headless=False
)
print("Environment 1 created successfully!")

print("Creating environment 2...")
print("Configuration for environment 2:")
print({
    'seed': 1,
    'task': "Jackal",
    'num_envs': num_envs,
    'sim_device': "cuda:0",
    'rl_device': "cuda:0",
    'graphics_device_id': 0,
    'headless': True
})
envs2 = isaacgymenvs.make(
	seed=1, 
	task="Jackal", 
	num_envs=num_envs, 
	sim_device="cuda:0",
	rl_device="cuda:0",
	graphics_device_id=0,
    headless=True
)
print("Environment 2 created successfully!")


import torch.nn as nn

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

class BasicAgent:
    def __init__(self, obs_dim, act_dim):
        self.model = PolicyValueNetwork(obs_dim, act_dim).cuda()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=3e-4)
        self.gamma = 0.99  # discount factor

    def get_action(self, obs):
        with torch.no_grad():
            mean, log_std, _ = self.model(obs)
            std = log_std.exp()
            normal = torch.distributions.Normal(mean, std)
            action = normal.sample()
        return action, normal.log_prob(action).sum(dim=-1, keepdim=True)

    def train(self, obs, action, reward, next_obs, done, log_prob):
        mean, log_std, values = self.model(obs)
        _, _, next_values = self.model(next_obs)
        
        # Calculate the expected value
        reward = reward.view(-1, 1)
        done = done.view(-1, 1)
        expected_values = reward + self.gamma * next_values * (1 - done.float())

        # Calculate the value loss
        value_loss = torch.nn.functional.mse_loss(values, expected_values.detach())

        # Calculate the policy loss using the advantage
        advantage = expected_values - values
        policy_loss = -(log_prob * advantage).mean()

        # Combine the losses
        loss = policy_loss + value_loss

        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)  # Gradient clipping
        self.optimizer.step()

cumulative_rewards1 = []
num_epochs = 1000
num_steps = 1000
agent1 = BasicAgent(envs1.observation_space.shape[0], envs1.action_space.shape[0])

for epoch in range(num_epochs):
    print(f"Starting epoch {epoch}...")
    start_time = time.time()

    obs1 = envs1.reset()
    total_reward1 = 0
    tensor_obs1 = obs1['obs']

    for step in range(num_steps):
        action1, log_prob1 = agent1.get_action(tensor_obs1)
        next_obs1, reward1, done1, _ = envs1.step(action1)
        total_reward1 += reward1.sum().item()
        agent1.train(tensor_obs1, action1, reward1, next_obs1['obs'], done1, log_prob1)
        tensor_obs1 = next_obs1['obs']

    elapsed_time = time.time() - start_time
    cumulative_rewards1.append(total_reward1)
    print(f"Epoch {epoch} completed! Average Reward: {total_reward1 / 200:.2f}, Time: {elapsed_time:.2f} seconds")
