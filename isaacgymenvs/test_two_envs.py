import sys
sys.path.append("/home/linjiw/Downloads/dcd-isaac/isaacgymenvs/")
sys.path.append("/home/linjiw/Downloads/dcd-isaac/")
sys.path.insert(0, './..')
import multiprocessing
import isaacgym
import isaacgymenvs
import torch
def train_agent(seed, headless):
    num_envs = 10

    env = isaacgymenvs.make(
        seed=seed, 
        task="Jackal", 
        num_envs=num_envs, 
        sim_device="cuda:0",
        rl_device="cuda:0",
        graphics_device_id=0,
        headless=headless
    )
    for i in range(100):
        random_actions = 2.0 * torch.rand((num_envs,) + env.action_space.shape, device = 'cuda:0') - 1.0
        env.step(random_actions)
        print(f"seed{seed} i")
    # agent = BasicAgent(env.observation_space.shape[0], env.action_space.shape[0])

    # Your training loop here
    # for epoch in range(num_epochs):
    #     ...

if __name__ == "__main__":
    # This is necessary for Windows
    agent1_process = multiprocessing.Process(target=train_agent, args=(0, False))
    agent2_process = multiprocessing.Process(target=train_agent, args=(1, True))

    agent1_process.start()
    agent2_process.start()

    agent1_process.join()
    agent2_process.join()
