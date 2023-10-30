import numpy as np
import torch
import torch.nn as nn
import torch.distributions as dist
from collections import defaultdict
import torch.nn.functional as F
# pip install matplotlib
import matplotlib.pyplot as plt

# class GridEnvironment:
#     # ... [rest of the code]

#     def display(self):
#         img = self.grid_to_image(self.grid)
#         plt.imshow(img)
#         plt.axis('off')  # Turn off axis numbers and ticks
#         plt.show()



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

    def grid_to_image(self, grid):
        image = np.zeros((15, 15, 3))
        image[grid == self.AGENT] = [0, 0, 255]  # Blue for agent
        image[grid == self.GOAL] = [0, 255, 0]  # Green for goal
        image[(grid == self.WALL) | (grid == self.OBSTACLE)] = [128, 128, 128]  # Gray for wall/obstacle
        return image

    def reset(self):
        self.grid = np.zeros((15, 15))
        self._initialize_walls()
        # Randomly place agent and goal
        empty_positions = np.argwhere(self.grid == self.EMPTY)
        agent_pos = empty_positions[np.random.choice(empty_positions.shape[0])]
        self.grid[agent_pos[0], agent_pos[1]] = self.AGENT
        goal_pos = empty_positions[np.random.choice(empty_positions.shape[0])]
        self.grid[goal_pos[0], goal_pos[1]] = self.GOAL
        return self.grid_to_image(self.grid)

    def step(self, action):
        # Convert action to position
        x, y = divmod(action, 15)
        self.place_object((x, y), self.OBSTACLE)
        reward = -1
        done = False
        return self.grid_to_image(self.grid), reward, done

    def place_object(self, position, object_type):
        x, y = position
        if self.grid[x, y] == self.EMPTY:
            self.grid[x, y] = object_type
        elif object_type == self.GOAL and self.grid[x, y] == self.AGENT:
            empty_positions = np.argwhere(self.grid == self.EMPTY)
            random_position = empty_positions[np.random.choice(empty_positions.shape[0])]
            self.grid[random_position[0], random_position[1]] = self.GOAL


    def display(self):
        img = self.grid_to_image(self.grid)
        plt.imshow(img)
        plt.axis('off')  # Turn off axis numbers and ticks
        plt.show()

    def save_grid_map(self, filename="grid_map.npy"):
        """Save the current grid state to a .npy file."""
        np.save(filename, self.grid)
def grid_to_image(observation):
    # Define the color mappings
    color_map = {
        GridEnvironment.EMPTY: [0, 0, 0],         # Black
        GridEnvironment.WALL: [128, 128, 128],   # Gray
        GridEnvironment.AGENT: [0, 0, 255],      # Blue
        GridEnvironment.GOAL: [0, 255, 0],       # Green
        GridEnvironment.OBSTACLE: [128, 128, 128] # Gray
    }
    
    # Convert the grid to a 3-channel image
    image = np.zeros((observation.shape[0], observation.shape[1], 3), dtype=np.uint8)
    for i in range(observation.shape[0]):
        for j in range(observation.shape[1]):
            image[i, j] = color_map[observation[i, j]]
    
    return image
import torch
import torch.nn as nn

class AdversaryPolicyNetwork(nn.Module):
    def __init__(self):
        super(AdversaryPolicyNetwork, self).__init__()
        self.conv = nn.Conv2d(3, 128, kernel_size=3, stride=1, padding=1)
        self.lstm = nn.LSTM(128*15*15 + 1 + 50, 256)  # +1 for t and +50 for z
        self.fc1 = nn.Linear(256, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, 169)  # Action space of 169

    def forward(self, x, t, z):
        x = self.conv(x)
        x = x.contiguous().view(x.size(0), -1)  # Make tensor contiguous and then flatten

        z_flattened = z.view(z.size(0), -1)

        #print(f"z_flattened {z_flattened.shape}")

        combined = torch.cat((x, t, z_flattened), dim=1).unsqueeze(0)
        #print(f'combined.shape {combined.shape}')
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
        self.lstm = nn.LSTM(128*15*15 + 1 + 50, 256)  # +1 for t and +50 for z
        self.fc1 = nn.Linear(256, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, 1)  # Single value output

    def forward(self, x, t, z):
        x = self.conv(x)
        x = x.contiguous().view(x.size(0), -1)  # Make tensor contiguous and then flatten

        z_flattened = z.view(z.size(0), -1)

        combined = torch.cat((x, t, z_flattened), dim=1).unsqueeze(0)
        # print(f'combined.shape {combined.shape}')
        x, _ = self.lstm(combined)
        x = x.squeeze(0)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

class Storage:
    def __init__(self):
        self.observations = []
        self.actions = []
        self.rewards = []
        self.masks = []
        self.values = []  # Added values attribute
        self.t = []  # Added t attribute
        self.z = []  # Added z attribute

    def insert(self, observation, action, reward, mask, t, z, value=None):
        self.observations.append(observation)
        self.actions.append(action)
        self.rewards.append(reward)
        self.masks.append(mask)
        self.t.append(t)
        self.z.append(z)
        if value is not None:
            self.values.append(value)

    def replace_all_rewards(self, new_reward):
        self.rewards = [new_reward for _ in self.rewards]

    def replace_final_return(self, final_return):
        self.rewards[-1] = final_return

    def compute_returns(self, next_value, gamma=0.995):
        returns = []
        R = next_value
        for reward in reversed(self.rewards):
            R = reward + gamma * R
            returns.insert(0, R)
        return returns

    def clear(self):
        self.observations.clear()
        self.actions.clear()
        self.rewards.clear()
        self.masks.clear()
        self.values.clear()  # Clear values as well
        self.t.clear()  # Clear t as well
        self.z.clear()  # Clear z as well


def rollout(adversary, env, num_steps, storage):
    observation = env.reset()
    for step in range(num_steps):
        # Sample a random vector z and set the timestep t
        # z = torch.randn(50)
        z = torch.randn(50).view(1, -1)
        t = torch.tensor([step], dtype=torch.float32)

        action = adversary.take_action(observation, t, z)
        # results = env.step(action)
        # print(results)

        # next_observation, _, done, _ = env.step(action)
        next_observation, reward, done = env.step(action)
        observation_tensor = torch.tensor(observation, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
        # print(observation_tensor.shape)

        observation_tensor = observation_tensor.permute(0, 3, 1, 2)  # Rearrange dimensions

        # observation_tensor = torch.tensor(observation, dtype=torch.float32).permute(0, 3, 1, 2)

        t_tensor = t.unsqueeze(1)  # Convert t from shape [batch_size] to [batch_size, 1]
        z_tensor = z.clone().detach().unsqueeze(1)  # Convert z from shape [batch_size] to [batch_size, 1]
        # print(f"")
        value = adversary.value_network(observation_tensor, t_tensor, z_tensor)

        # value = adversary.value_network(torch.tensor(observation, dtype=torch.float32).unsqueeze(0))
        storage.values.append(value.item())

        
        mask = 0 if done else 1
        # Placeholder reward, will be replaced later
        reward = 0

        # storage.insert(observation, action, reward, mask)
        # print(f"t_tensor inserted {t_tensor.shape}")
        storage.insert(observation, action, reward, mask, t_tensor, z_tensor.tolist())

        observation = next_observation

        if done:
            observation = env.reset()

class Adversary:
    def __init__(self):
        self.policy_network = AdversaryPolicyNetwork()
        self.value_network = AdversaryValueNetwork()
        self.optimizer = torch.optim.Adam(list(self.policy_network.parameters()) + list(self.value_network.parameters()), lr=0.0001)

    def take_action(self, observation, t, z):
        observation_tensor = torch.tensor(observation, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
        # print(observation_tensor.shape)

        observation_tensor = observation_tensor.permute(0, 3, 1, 2)  # Rearrange dimensions

        # observation_tensor = torch.tensor(observation, dtype=torch.float32).permute(0, 3, 1, 2)

        t_tensor = t.unsqueeze(1)  # Convert t from shape [batch_size] to [batch_size, 1]
        z_tensor = z.clone().detach().unsqueeze(1)  # Convert z from shape [batch_size] to [batch_size, 1]
        
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

    def save_model(self, path):
        """
        Save the model parameters to the specified file.
        
        Args:
        - path (str): The path to the file where model parameters should be saved.
        """
        model_data = {
            'policy_state_dict': self.policy_network.state_dict(),
            'value_state_dict': self.value_network.state_dict(),
        }
        torch.save(model_data, path)

    def load_model(self, path):
        """
        Load the model parameters from the specified file.
        
        Args:
        - path (str): The path to the file from which model parameters should be loaded.
        """
        model_data = torch.load(path)
        self.policy_network.load_state_dict(model_data['policy_state_dict'])
        self.value_network.load_state_dict(model_data['value_state_dict'])



def compute_regret(agent_performance):
    # For now, return a random number as the regret
    return np.random.rand()

# def 
# def 

def train_adversary(adversary, env, agent_performance, num_epochs, gamma=0.995, clip_epsilon=0.2, value_coeff=0.5, entropy_coeff=0.01):
    optimizer = torch.optim.Adam(adversary.policy_network.parameters(), lr=0.0001)
    value_optimizer = torch.optim.Adam(adversary.value_network.parameters(), lr=0.0001)
    storage = Storage()

    for epoch in range(num_epochs):
        rollout(adversary, env, num_steps=50, storage=storage)
        # env.display()
        # Compute the regret based on agent performance

        regret = compute_regret(agent_performance)
        storage.replace_all_rewards(regret)
        
        # Convert lists to tensors
        observations = torch.tensor(storage.observations, dtype=torch.float32)
        observations = torch.tensor(storage.observations, dtype=torch.float32).permute(0, 3, 1, 2)

        actions = torch.tensor(storage.actions, dtype=torch.int64)
        rewards = torch.tensor(storage.rewards, dtype=torch.float32)
        old_values = torch.tensor(storage.values, dtype=torch.float32)  # Use values from storage
        t = torch.tensor(storage.t, dtype=torch.float32)  # Assuming you have a t list in storage
        z = torch.tensor(storage.z, dtype=torch.float32)  # Assuming you have a z list in storage
        t = t.unsqueeze(1)

        # Compute returns and advantages
        returns = torch.zeros_like(rewards)
        advantages = torch.zeros_like(rewards)
        R = 0
        for i in reversed(range(len(rewards))):
            R = rewards[i] + gamma * R
            returns[i] = R
            advantages[i] = R - old_values[i]

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)

        # PPO update
        print(f"PPO------------------")
    for _ in range(4):  # number of PPO epochs
        logits = adversary.policy_network(observations, t, z)
        probs = F.softmax(logits, dim=1)
        log_probs = F.log_softmax(logits, dim=1)
        dist_entropy = -(probs * log_probs).sum(-1).mean()
        
        action_log_probs = log_probs.gather(1, actions.unsqueeze(-1)).squeeze(-1)
        old_action_log_probs = action_log_probs.detach()

        ratio = torch.exp(action_log_probs - old_action_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantages

        # Policy loss
        policy_loss = -torch.min(surr1, surr2).mean()

        # Value loss
        current_values = adversary.value_network(observations, t, z)
        value_loss = 0.5 * (returns - current_values).pow(2).mean()

        # Total loss
        loss = policy_loss + value_coeff * value_loss - entropy_coeff * dist_entropy.mean()

        optimizer.zero_grad()
        loss.backward(retain_graph=True)

        optimizer.step()

        value_optimizer.zero_grad()
        value_loss.backward()
        value_optimizer.step()

adversary = Adversary()
env = GridEnvironment()
agent_performance = 0  # Placeholder
train_adversary(adversary, env, agent_performance, num_epochs=100)