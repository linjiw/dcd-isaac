import numpy as np
import torch
import torch.nn as nn

class GridEnvironment:
    EMPTY = 0
    WALL = 1
    AGENT = 2
    GOAL = 3
    OBSTACLE = 4

    def __init__(self):
        self.grid = np.zeros((15, 15))
        self._initialize_walls()

    def _initialize_walls(self):
        # Set walls around the edges
        self.grid[0, :] = self.WALL
        self.grid[-1, :] = self.WALL
        self.grid[:, 0] = self.WALL
        self.grid[:, -1] = self.WALL

    def reset(self):
        self.grid = np.zeros((15, 15))
        self._initialize_walls()
        # Randomly place agent and goal
        empty_positions = np.argwhere(self.grid == self.EMPTY)
        agent_pos = empty_positions[np.random.choice(empty_positions.shape[0])]
        self.grid[agent_pos[0], agent_pos[1]] = self.AGENT
        goal_pos = empty_positions[np.random.choice(empty_positions.shape[0])]
        self.grid[goal_pos[0], goal_pos[1]] = self.GOAL
        return self.grid

    def step(self, action):
        # Convert action to position
        x, y = divmod(action, 15)
        self.place_object((x, y), self.OBSTACLE)
        # For now, we'll assume a constant reward and that the episode is never done
        # This can be modified based on specific criteria
        reward = -1
        done = False
        return self.grid, reward, done

    def place_object(self, position, object_type):
        x, y = position
        if self.grid[x, y] == self.EMPTY:  # Only place if the tile is empty
            self.grid[x, y] = object_type
        elif object_type == self.GOAL and self.grid[x, y] == self.AGENT:
            # If trying to place the goal on top of the agent, place the goal randomly
            empty_positions = np.argwhere(self.grid == self.EMPTY)
            random_position = empty_positions[np.random.choice(empty_positions.shape[0])]
            self.grid[random_position[0], random_position[1]] = self.GOAL

    def display(self):
        # For visualization purposes
        for row in self.grid:
            print(''.join(['#' if cell == self.WALL else 
                           'A' if cell == self.AGENT else 
                           'G' if cell == self.GOAL else 
                           'O' if cell == self.OBSTACLE else 
                           ' ' for cell in row]))

# Adversary
import torch
import torch.nn as nn

class AdversaryPolicyNetwork(nn.Module):
    def __init__(self):
        super(AdversaryPolicyNetwork, self).__init__()
        self.conv = nn.Conv2d(3, 128, kernel_size=3, stride=1, padding=1)
        self.lstm = nn.LSTM(128 + 1 + 50, 256)  # +1 for t and +50 for z
        self.fc1 = nn.Linear(256, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, 169)  # Action space of 169

    def forward(self, x, t, z):
        x = self.conv(x)
        combined = torch.cat((x.view(x.size(0), -1), t, z), dim=1).unsqueeze(0)
        x, _ = self.lstm(combined)
        x = x.squeeze(0)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

class AdversaryValueNetwork(nn.Module):
    def __init__(self):
        super(AdversaryValueNetwork, self).__init__()
        self.conv = nn.Conv2d(3, 128, kernel_size=3, stride=1, padding=1)
        self.lstm = nn.LSTM(128 + 1 + 50, 256)  # +1 for t and +50 for z
        self.fc1 = nn.Linear(256, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, 1)  # Single value output

    def forward(self, x, t, z):
        x = self.conv(x)
        combined = torch.cat((x.view(x.size(0), -1), t, z), dim=1).unsqueeze(0)
        x, _ = self.lstm(combined)
        x = x.squeeze(0)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


class Adversary:
    def __init__(self):
        self.policy_network = AdversaryPolicyNetwork()
        self.value_network = AdversaryValueNetwork()
        self.optimizer = torch.optim.Adam(list(self.policy_network.parameters()) + list(self.value_network.parameters()), lr=0.0001)

    def take_action(self, observation, t, z):
        observation_tensor = torch.tensor(observation, dtype=torch.float32).unsqueeze(0)
        t_tensor = torch.tensor([t], dtype=torch.float32)
        z_tensor = torch.tensor(z, dtype=torch.float32)
        
        action_probs = self.policy_network(observation_tensor, t_tensor, z_tensor)
        action = torch.argmax(action_probs, dim=1).item()
        return action

    def compute_loss(self, rewards, values, log_probs):
        # Compute the loss for PPO here
        pass

    def update(self, rewards, values, log_probs):
        loss = self.compute_loss(rewards, values, log_probs)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def rollout(self, env, num_steps):
        # ... (same as before)

# # Agent
# class AgentNN(nn.Module):
#     def __init__(self):
#         super(AgentNN, self).__init__()
#         self.conv = nn.Conv2d(3, 128, kernel_size=3, stride=1, padding=1)
#         self.lstm = nn.LSTM(128, 256)
#         self.fc1 = nn.Linear(256, 32)
#         self.fc2 = nn.Linear(32, 32)
#         self.fc3 = nn.Linear(32, 4)  # Assuming 4 possible movement actions

#     def forward(self, x):
#         x = self.conv(x)
#         x, _ = self.lstm(x.view(x.size(0), -1).unsqueeze(0))
#         x = x.squeeze(0)
#         x = self.fc1(x)
#         x = self.fc2(x)
#         x = self.fc3(x)
#         return x

# class Agent:
#     def __init__(self):
#         self.model = AgentNN()

#     def take_action(self, observation):
#         observation_tensor = torch.tensor(observation, dtype=torch.float32).unsqueeze(0)
#         action_probs = self.model(observation_tensor)
#         action = torch.argmax(action_probs, dim=1).item()
#         return action

# Note: This is a basic structure. Training methods, loss functions, optimizers, and other functionalities need to be added.
env = GridEnvironment()
env.place_object((7, 7), GridEnvironment.AGENT)
env.place_object((8, 8), GridEnvironment.GOAL)
env.display()
