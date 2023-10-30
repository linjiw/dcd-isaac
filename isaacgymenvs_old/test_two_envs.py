
import wandb
import sys
sys.path.append("/home/linjiw/Downloads/dcd-isaac/isaacgymenvs/")
sys.path.append("/home/linjiw/Downloads/dcd-isaac/")
sys.path.insert(0, './..')
import multiprocessing
import isaacgym
import isaacgymenvs
import torch
from isaac_train_single_agent import BasicAgent
from isaac_train_single_agent import PolicyValueNetwork
import time
from tqdm import tqdm

import numpy as np
class Trainer:
    def __init__(self, agent_name, seed, num_envs, headless, info_queue):

        self.agent_name = agent_name
        self.env = isaacgymenvs.make(
            seed=seed, 
            task="Jackal", 
            num_envs=num_envs, 
            sim_device="cuda:0",
            rl_device="cuda:0",
            graphics_device_id=0,
            headless=headless
        )
        # print(f"env.action_space.low: {self.env.action_space.low}")
        # print(f"env.action_space.high: {self.env.action_space.high}")

        self.num_envs = num_envs
        self.agent = BasicAgent(self.env.observation_space.shape[0], self.env.action_space.shape[0],self.env.action_space.low,self.env.action_space.high,device="cuda:0")
        self.info_queue = info_queue
        print(f"Trainer for {self.agent_name} created successfully!")
    
    def agent_rollout(self, num_steps):
        obs = self.env.reset()
        rollout_data = {
            'observations': [],
            'actions': [],
            'rewards': [],
            'log_probs': []
        }

        for step in range(num_steps):
            tensor_obs = obs['obs']
            action, log_prob = self.agent.get_action(tensor_obs)

            norm_actions_tensor = self.agent.normalizeAction(action)
            action = norm_actions_tensor
            next_obs, reward, done, _ = self.env.step(action)

            rollout_data['observations'].append(tensor_obs)
            rollout_data['actions'].append(action)
            # rollout_data['rewards'].append(torch.tensor(reward))  # Ensure reward is a tensor
            rollout_data['rewards'].append(reward.clone().detach())

            rollout_data['log_probs'].append(log_prob)

            obs = next_obs
        wandb.log({
            f"{self.agent_name} Rollout Mean Reward": torch.stack(rollout_data['rewards']).mean().item()
        })
        print(f"Rollout completed for {self.agent_name}!")
        return rollout_data

 

    def train(self, num_epochs, num_steps, sync_interval):
        cumulative_rewards = []
        pbar = tqdm(range(num_epochs), desc="Training Progress", ncols=100)  # Capture the tqdm object in pbar
        for epoch in pbar:
        # for epoch in tqdm(range(num_epochs), desc="Training Progress", ncols=100):
            obs = self.env.reset()
            tensor_obs = obs['obs']  # Extract tensor from dictionary
            total_rewards = torch.zeros(self.num_envs, device='cuda:0')
            start_time = time.time()
            policy_losses = []
            value_losses = []
            for step in range(num_steps):
                action, log_prob = self.agent.get_action(tensor_obs)

                norm_actions_tensor = self.agent.normalizeAction(action)
                action = norm_actions_tensor
                next_obs, reward, done, _ = self.env.step(action)
                tensor_next_obs = next_obs['obs']  # Extract tensor from dictionary

                # Update the agent immediately
                policy_loss, value_loss = self.agent.train(tensor_obs, action, reward, tensor_next_obs, done, log_prob)
                policy_losses.append(policy_loss)
                value_losses.append(value_loss)
                total_rewards += reward.clone().detach()
                tensor_obs = tensor_next_obs  # Update the tensor observation for the next step

            elapsed_time = time.time() - start_time
            cumulative_rewards.append(total_rewards)
            average_reward = total_rewards.mean().item()
            cumulative_rewards_tensor = torch.cat(cumulative_rewards)
            mean_return = cumulative_rewards_tensor.mean().item()
            max_return = cumulative_rewards_tensor.max().item()

            # Update tqdm description with training information
            pbar.set_description(f"Epoch {epoch}/{num_epochs} | Avg Policy Loss: {np.mean(policy_losses):.4f} | Avg Value Loss: {np.mean(value_losses):.4f} | Avg Reward: {average_reward:.2f} | Time: {elapsed_time:.2f}s")
            pbar.refresh()

            wandb.log({
                f"{self.agent_name} Average Policy Loss": np.mean(policy_losses),
                f"{self.agent_name} Average Value Loss": np.mean(value_losses),
                f"{self.agent_name} Average Reward": average_reward,
                f"{self.agent_name} Mean Return": mean_return,
                f"{self.agent_name} Max Return": max_return,
                f"{self.agent_name} Epoch Time": elapsed_time
            })

            # if self.agent_name == "antagonist":
            #     self.info_queue.put({'epoch': epoch, 'mean_return': mean_return, 'max_return': max_return})

            # if epoch % sync_interval == 0 and self.agent_name == "protagonist":
            #     while True:
            #         adversary_info = self.info_queue.get()
            #         print(f"waiting for another agent to finish epoch {adversary_info['epoch']}...")
            #         if adversary_info['epoch'] == epoch:
            #             env_return = torch.max(torch.tensor(adversary_info['max_return']), torch.tensor(mean_return)) - torch.tensor(mean_return)
            #             print(f"Regret between Antagonist and Protagonist: {env_return.item()}")
            #             wandb.log({
            #                 "Protagonist env_return": env_return.item()
            #             })
            #             break


def train_agent(agent_name, seed,num_envs, headless, info_queue, num_epochs, num_steps, sync_interval):
    # wandb.init(project="isaacgymenvs", name=f"{agent_name}_training")

    trainer = Trainer(agent_name, seed,num_envs, headless, info_queue)
    trainer.train(num_epochs, num_steps, sync_interval)
    # wandb.finish()
if __name__ == "__main__":
    wandb.init(project="isaacgymenvs", name="two agent training")

    info_queue = multiprocessing.Queue()

    num_epochs = 10000
    num_steps = 50
    sync_interval = 20
    num_envs = 50
    protagonist_process = multiprocessing.Process(target=train_agent, args=("protagonist", 0,num_envs, False, info_queue, num_epochs, num_steps, sync_interval))
    # antagonist_process = multiprocessing.Process(target=train_agent, args=("antagonist", 1,num_envs, False, info_queue, num_epochs, num_steps, sync_interval))

    protagonist_process.start()
    # antagonist_process.start()

    protagonist_process.join()
    # antagonist_process.join()
    wandb.finish()