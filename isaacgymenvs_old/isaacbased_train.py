import sys
sys.path.append("/home/linjiw/Downloads/dcd-isaac/isaacgymenvs/")
sys.path.append("/home/linjiw/Downloads/dcd-isaac/")
sys.path.insert(0, './..')

import isaacgym
import isaacgymenvs
import torch
import time
import numpy as np


num_envs = 10

print("Creating environment 1...")
print("Configuration for environment 1:")
print({
    'seed': 0,
    'task': "Jackal",
    'num_envs': num_envs,
    'sim_device': "cuda:0",
    'rl_device': "cuda:0",
    'graphics_device_id': 0,
    'headless': True
})

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
# print("Creating environment 2...")
# print("Configuration for environment 2:")
# print({
#     'seed': 1,
#     'task': "Jackal",
#     'num_envs': num_envs,
#     'sim_device': "cuda:0",
#     'rl_device': "cuda:0",
#     'graphics_device_id': 0,
#     'headless': True
# })
# envs2 = isaacgymenvs.make(
# 	seed=1, 
# 	task="Jackal", 
# 	num_envs=num_envs, 
# 	sim_device="cuda:0",
# 	rl_device="cuda:0",
# 	graphics_device_id=0,
#     headless=True
# )
# print("Environment 2 created successfully!")

import torch.nn as nn

# class PolicyValueNetwork(nn.Module):
#     def __init__(self, obs_dim, act_dim):
#         super(PolicyValueNetwork, self).__init__()
#         self.fc1 = nn.Linear(obs_dim, 64)
#         self.fc2 = nn.Linear(64, 64)
#         self.policy_head = nn.Linear(64, act_dim)
#         self.value_head = nn.Linear(64, 1)

#     def forward(self, x):
#         x = torch.relu(self.fc1(x))
#         x = torch.relu(self.fc2(x))
#         # return torch.tanh(self.policy_head(x)), self.value_head(x)
#         return torch.softmax(self.policy_head(x), dim=-1), self.value_head(x)
class PolicyValueNetwork(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super(PolicyValueNetwork, self).__init__()
        self.fc1 = nn.Linear(obs_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.mean_head = nn.Linear(64, act_dim)
        self.log_std_head = nn.Linear(64, act_dim)
        self.value_head = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        mean = self.mean_head(x)
        log_std = self.log_std_head(x)
        values = self.value_head(x)
        return mean, log_std, values

class BasicAgent:
    def __init__(self, obs_dim, act_dim):
        self.model = PolicyValueNetwork(obs_dim, act_dim).cuda()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        self.gamma = 0.99  # discount factor

    # def get_action(self, obs):
    #     with torch.no_grad():
    #         action_probs, _ = self.model(obs)
    #         print(f"action_probs {action_probs}")

    #         action = torch.multinomial(action_probs, 1).squeeze()
    #     return action
    # def get_action(self, obs):
    #     with torch.no_grad():
    #         action_probs, _ = self.model(obs)
    #         print("Action probabilities shape:", action_probs.shape)
    #         print("Action probabilities values:", action_probs)
    #         action = torch.multinomial(action_probs, 2)
    #     return action
    
    def get_action(self, obs):
        with torch.no_grad():
            mean, log_std, _ = self.model(obs)
            std = log_std.exp()
            normal = torch.distributions.Normal(mean, std)
            action = normal.sample()
        return action, normal.log_prob(action).sum(dim=-1, keepdim=True)


    # def train(self, obs, action, reward, next_obs, done):
    #     print("Training agent...")
    #     # Forward pass
    #     action_probs, values = self.model(obs)
    #     _, next_values = self.model(next_obs)

    #     # Calculate the expected value
    #     reward = reward.view(-1, 1)
    #     done = done.view(-1, 1)
    #     expected_values = reward + self.gamma * next_values * (1 - done.float())
    #     print(f"reward.shape {reward.shape}")
    #     print(f"next_values.shape {next_values.shape}")
    #     print(f"done.shape {done.shape}")
    #     # expected_values = expected_values.view(values.shape)

    #     # Calculate the value loss
    #     print("values.shape:", values.shape)
    #     print("expected_values.shape:", expected_values.shape)
    #     value_loss = torch.nn.functional.mse_loss(values, expected_values.detach())

    #     # Calculate the policy loss
    #     print("action_probs.shape:", action_probs.shape)
    #     print("action.unsqueeze(-1).shape:", action.unsqueeze(-1).shape)

    #     action_log_probs = torch.log(action_probs.gather(1, action.unsqueeze(-1)))
    #     advantage = expected_values - values
    #     policy_loss = -(action_log_probs * advantage).mean()

    #     # Combine the losses
    #     loss = policy_loss + value_loss

    #     # Backpropagation
    #     self.optimizer.zero_grad()
    #     loss.backward()
    #     self.optimizer.step()
    #     print("Agent trained successfully!")

    # def train(self, obs, action, reward, next_obs, done):
    def train(self, obs, action, reward, next_obs, done, log_prob):

        # print("Training agent...")
        mean, log_std, values = self.model(obs)
        _, _, next_values = self.model(next_obs)
        
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        log_prob = normal.log_prob(action).sum(dim=-1, keepdim=True)
        
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
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)  # You can adjust the max_norm value

        self.optimizer.step()   
        # print("Agent trained successfully!")


num_epochs = 1000
num_steps = 200
sync_interval = 10  # Sync every 10 epochs
print(f"action_space {envs1.action_space.shape}")
agent1 = BasicAgent(envs1.observation_space.shape[0], envs1.action_space.shape[0])
# agent2 = BasicAgent(envs2.observation_space.shape[0], envs2.action_space.shape[0])

def synchronize_agents(agent1, agent2, rewards1, rewards2):
    # Calculate regret based on the difference in cumulative rewards
    print("Synchronizing agents...")
    regret = sum(rewards2) - sum(rewards1)
    
    print(f"Regret between Agent 2 and Agent 1: {regret}")

    # If you want to take any action based on regret, you can do it here.
    # For instance, if regret is too high, you might want to copy weights from the better agent to the other.
    # if regret > SOME_THRESHOLD:
    #     agent1.model.load_state_dict(agent2.model.state_dict())


cumulative_rewards1 = []
cumulative_rewards2 = []

for epoch in range(num_epochs):
    print(f"Starting epoch {epoch}...")
    start_time = time.time()

    obs1 = envs1.reset()
    # obs2 = envs2.reset()
    print(f"obs1 {obs1}")
    
    total_reward1 = 0
    total_reward2 = 0
    tensor_obs1 = obs1['obs']

    for step in range(num_steps):
        # print("Agent 1 taking action...")
        if torch.isnan(tensor_obs1).any() or torch.isinf(tensor_obs1).any():
            print("Invalid observation:", tensor_obs1)
        # action1 = agent1.get_action(tensor_obs1)
        action1, log_prob1 = agent1.get_action(tensor_obs1)
        # print("Agent 1 action taken!")
        # print("Agent 2 taking action...")

        # action2 = agent2.get_action(obs2)
        # print("Agent 2 action taken!")
        # print("Stepping through environment 1...")
        # print(f"action1 {action1}")
        next_obs1, reward1, done1, _ = envs1.step(action1)
        # print("Stepped through environment 1!")
        # print("Stepping through environment 2...")

        # # next_obs2, reward2, done2, _ = envs2.step(action2)
        # print("Stepped through environment 2!")

        total_reward1 += reward1.sum().item()
        # total_reward2 += reward2.sum().item()
        # print(f"obs1 {obs1}")

        # Train the agents
        # print(f"tensor_obs1 {tensor_obs1}")
        # print(f"next_obs1 {next_obs1}")

        # agent1.train(tensor_obs1, action1, reward1, next_obs1['obs'], done1)
        agent1.train(tensor_obs1, action1, reward1, next_obs1['obs'], done1, log_prob1)

        # agent2.train(obs2, action2, reward2, next_obs2, done2)
        
        tensor_obs1 = next_obs1['obs']
        # print(f"obs1 {obs1}")
        # obs2 = next_obs2

    cumulative_rewards1.append(total_reward1)
    cumulative_rewards2.append(total_reward2)
    elapsed_time = time.time() - start_time
    print(f"Epoch {epoch} completed! Average Reward: {total_reward1 / num_steps:.2f}, Time: {elapsed_time:.2f} seconds")

    # Synchronize agents after certain epochs
    # if epoch % sync_interval == 0:
    #     synchronize_agents(agent1, agent2, cumulative_rewards1, cumulative_rewards2)
    # print(f"Epoch {epoch} completed!")

