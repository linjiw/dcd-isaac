# import sys
# sys.path.append("/home/linjiw/Downloads/dcd-isaac/isaacgymenvs/")
# sys.path.append("/home/linjiw/Downloads/dcd-isaac/")
# sys.path.insert(0, './..')
# import multiprocessing
# import isaacgym
# import isaacgymenvs
# import torch
# from isaac_train_single_agent import BasicAgent
# from isaac_train_single_agent import PolicyValueNetwork
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
#         print(f"seed{seed} is working") 
#     agent = BasicAgent(env.observation_space.shape[0], env.action_space.shape[0])

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
import numpy as np
class Trainer:
    def __init__(self, agent_name, seed, num_envs, headless, info_queue):

        self.agent_name = agent_name
        self.env = isaacgymenvs.make(
            seed=seed, 
            task="new_jackal", 
            num_envs=num_envs, 
            sim_device="cuda:0",
            rl_device="cuda:0",
            graphics_device_id=0,
            headless=headless
        )
        self.num_envs = num_envs
        self.agent = BasicAgent(self.env.observation_space.shape[0], self.env.action_space.shape[0])
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

        for epoch in range(num_epochs):
            print(f"Starting epoch {epoch} for {self.agent_name}...")
            start_time = time.time()

            rollout_data = self.agent_rollout(num_steps)
            total_reward = sum(rollout_data['rewards'])

            # Update agent using the collected rollout data
            self.agent.update_from_rollout(rollout_data)

            elapsed_time = time.time() - start_time
            cumulative_rewards.append(total_reward)

            average_reward = (total_reward.sum() / (num_steps * self.num_envs)).item()
            cumulative_rewards_tensor = torch.cat(cumulative_rewards)

            print(f"cumulative_rewards {cumulative_rewards}")

            # if len(cumulative_rewards) == 1:
            #     if isinstance(cumulative_rewards[0], torch.Tensor):
            #         cumulative_rewards_tensor = cumulative_rewards[0]
            #     else:
            #         cumulative_rewards_tensor = torch.tensor(cumulative_rewards[0])
            # else:
            #     cumulative_rewards_tensor = torch.tensor(cumulative_rewards)

            mean_return = cumulative_rewards_tensor.mean().item()
            max_return = cumulative_rewards_tensor.max().item()

            print(f"Epoch {epoch} completed for {self.agent_name}! Average Reward: {average_reward:.2f}, Time: {elapsed_time:.2f} seconds")
            # wandb.log({"Average Reward": average_reward, "Mean Return": mean_return, "Max Return": max_return})
            wandb.log({
                f"{self.agent_name} Average Reward": average_reward,
                f"{self.agent_name} Mean Return": mean_return,
                f"{self.agent_name} Max Return": max_return,
                f"{self.agent_name} Epoch Time": elapsed_time
            })
            if self.agent_name == "antagonist":
                self.info_queue.put({'epoch': epoch, 'mean_return': mean_return, 'max_return': max_return})

            if epoch % sync_interval == 0 and self.agent_name == "protagonist":
                while True:
                    adversary_info = self.info_queue.get()
                    print(f"waiting for another agent to finish epoch {adversary_info['epoch']}...")
                    if adversary_info['epoch'] == epoch:
                        env_return = torch.max(torch.tensor(adversary_info['max_return']), torch.tensor(mean_return)) - torch.tensor(mean_return)
                        print(f"Regret between Antagonist and Protagonist: {env_return.item()}")
                        wandb.log({
                        "Protagonist env_return": env_return.item()
                    })
                        break



def train_agent(agent_name, seed,num_envs, headless, info_queue, num_epochs, num_steps, sync_interval):
    # wandb.init(project="isaacgymenvs", name=f"{agent_name}_training")

    trainer = Trainer(agent_name, seed,num_envs, headless, info_queue)
    trainer.train(num_epochs, num_steps, sync_interval)
    # wandb.finish()
if __name__ == "__main__":
    wandb.init(project="isaacgymenvs", name="two agent training")

    info_queue = multiprocessing.Queue()

    num_epochs = 1000
    num_steps = 100
    sync_interval = 1
    num_envs = 20
    protagonist_process = multiprocessing.Process(target=train_agent, args=("protagonist", 0,num_envs, True, info_queue, num_epochs, num_steps, sync_interval))
    antagonist_process = multiprocessing.Process(target=train_agent, args=("antagonist", 1,num_envs, False, info_queue, num_epochs, num_steps, sync_interval))

    protagonist_process.start()
    antagonist_process.start()

    protagonist_process.join()
    antagonist_process.join()
    wandb.finish()