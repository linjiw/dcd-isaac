# import sys
# sys.path.append("/home/linjiw/Downloads/dcd-isaac/isaacgymenvs/")
# sys.path.append("/home/linjiw/Downloads/dcd-isaac/")
# sys.path.insert(0, './..')
# import multiprocessing
# import isaacgym
# import isaacgymenvs
# import torch
# def train_agent(seed, headless):
#     num_envs = 10

#     env = isaacgymenvs.make(
#         seed=seed, 
#         task="Jackal", 
#         num_envs=num_envs, 
#         sim_device="cuda:0",
#         rl_device="cuda:0",
#         graphics_device_id=0,
#         headless=headless
#     )
#     for i in range(100):
#         random_actions = 2.0 * torch.rand((num_envs,) + env.action_space.shape, device = 'cuda:0') - 1.0
#         env.step(random_actions)
#         print(f"seed{seed} i")
#     # agent = BasicAgent(env.observation_space.shape[0], env.action_space.shape[0])

#     # Your training loop here
#     # for epoch in range(num_epochs):
#     #     ...

# if __name__ == "__main__":
#     # This is necessary for Windows
#     agent1_process = multiprocessing.Process(target=train_agent, args=(0, False))
#     agent2_process = multiprocessing.Process(target=train_agent, args=(1, True))

#     agent1_process.start()
#     agent2_process.start()

#     agent1_process.join()
#     agent2_process.join()


# import multiprocessing

# import isaacgym
# import isaacgymenvs
# import torch
# import time
# import numpy as np
# from isaac_train_single_agent import BasicAgent
# from isaac_train_single_agent import PolicyValueNetwork


# class Trainer:
#     def __init__(self, seed, headless, info_queue):
#         self.env = isaacgymenvs.make(
#             seed=seed, 
#             task="Jackal", 
#             num_envs=10, 
#             sim_device="cuda:0",
#             rl_device="cuda:0",
#             graphics_device_id=0,
#             headless=headless
#         )
#         self.agent = BasicAgent(self.env.observation_space.shape[0], self.env.action_space.shape[0])
#         self.reward_queue = reward_queue
#         self.info_queue = info_queue

#     def train(self, num_epochs, sync_interval):
#         for epoch in range(num_epochs):
#             # ... training logic ...

#             # After each epoch, put the mean and max rewards into the queue
#             self.info_queue.put({'mean_return': mean_reward, 'max_return': max_reward})

#             # If it's the synchronization epoch
#             if epoch % sync_interval == 0:
#                 # Wait for info from both agents
#                 agent_info = self.info_queue.get()
#                 adversary_agent_info = self.info_queue.get()

#                 # Compute regret
#                 env_return = torch.max(adversary_agent_info['max_return'] - agent_info['mean_return'], \
#                     torch.zeros_like(agent_info['mean_return']))

#                 # # Normalize regret
#                 # self.env_return_rms.update(env_return.flatten().cpu().numpy())
#                 # env_return /= np.sqrt(self.env_return_rms.var + 1e-8)

#                 # # Clip regret
#                 # clip_max_abs = 1.0  # Set this to your desired value
#                 # env_return = env_return.clamp(-clip_max_abs, clip_max_abs)

#                 print(f"Regret between Agent 2 and Agent 1: {env_return.item()}")


# def train_agent(seed, headless, reward_queue, num_epochs, sync_interval):
#     trainer = Trainer(seed, headless, reward_queue)
#     trainer.train(num_epochs, sync_interval)
import multiprocessing

import isaacgym
import isaacgymenvs
import torch
import time
import numpy as np
from isaac_train_single_agent import BasicAgent
from isaac_train_single_agent import PolicyValueNetwork

class Trainer:
    def __init__(self, agent_name, seed, headless, info_queue):

        self.agent_name = agent_name
        self.env = isaacgymenvs.make(
            seed=seed, 
            task="Jackal", 
            num_envs=10, 
            sim_device="cuda:0",
            rl_device="cuda:0",
            graphics_device_id=0,
            headless=headless
        )
        self.agent = BasicAgent(self.env.observation_space.shape[0], self.env.action_space.shape[0])
        self.info_queue = info_queue

    def train(self, num_epochs, num_steps, sync_interval):
        cumulative_rewards = []

        for epoch in range(num_epochs):
            print(f"Starting epoch {epoch} for {self.agent_name}...")
            start_time = time.time()

            obs = self.env.reset()
            total_reward = 0
            tensor_obs = obs['obs']

            for step in range(num_steps):
                action, log_prob = self.agent.get_action(tensor_obs)
                next_obs, reward, done, _ = self.env.step(action)
                total_reward += reward.sum().item()
                self.agent.train(tensor_obs, action, reward, next_obs['obs'], done, log_prob)
                tensor_obs = next_obs['obs']

            elapsed_time = time.time() - start_time
            cumulative_rewards.append(total_reward)
            print(f"Epoch {epoch} completed for {self.agent_name}! Average Reward: {total_reward / num_steps:.2f}, Time: {elapsed_time:.2f} seconds")

            # After each epoch, put the mean and max rewards into the queue
            # After each epoch, put the mean, max rewards, and epoch number into the queue
            if self.agent_name == "antagonist":
                self.info_queue.put({'epoch': epoch, 'mean_return': np.mean(cumulative_rewards), 'max_return': np.max(cumulative_rewards)})

            # If it's the synchronization epoch and the agent is the protagonist
            if epoch % sync_interval == 0 and self.agent_name == "protagonist":
                while True:
                    adversary_info = self.info_queue.get()
                    if adversary_info['epoch'] == epoch:
                        # Compute regret
                        env_return = torch.max(adversary_info['max_return'] - np.mean(cumulative_rewards), \
                            torch.zeros_like(np.mean(cumulative_rewards)))
                        print(f"Regret between Antagonist and Protagonist: {env_return.item()}")
                        break

def train_agent(agent_name, seed, headless, info_queue, num_epochs, num_steps, sync_interval):
    trainer = Trainer(agent_name, seed, headless, info_queue)
    trainer.train(num_epochs, num_steps, sync_interval)

if __name__ == "__main__":
    info_queue = multiprocessing.Queue()

    num_epochs = 1000
    num_steps = 200
    sync_interval = 10

    protagonist_process = multiprocessing.Process(target=train_agent, args=("protagonist", 0, False, info_queue, num_epochs, num_steps, sync_interval))
    antagonist_process = multiprocessing.Process(target=train_agent, args=("antagonist", 1, True, info_queue, num_epochs, num_steps, sync_interval))

    protagonist_process.start()
    antagonist_process.start()

    protagonist_process.join()
    antagonist_process.join()
