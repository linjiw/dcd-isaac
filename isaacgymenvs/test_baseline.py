

# env = gym.make('CartPole-v1')
import sys
# sys.path.append("/home/linjiw/Downloads/isaacgym-jackal/")
sys.path.insert(0, './..')

import isaacgym
import os
from rl_games.common import env_configurations, vecenv
from rl_games.torch_runner import Runner
from rl_games.algos_torch import model_builder
from isaacgymenvs.learning import amp_continuous, amp_players, amp_models, amp_network_builder
from isaacgymenvs.utils.rlgames_utils import RLGPUAlgoObserver
from stable_baselines3 import PPO
import isaacgymenvs
import gym
def create_env(seed, task_name, num_envs, sim_device, rl_device, graphics_device_id):
    envs = isaacgymenvs.make(
        seed, task_name, num_envs, sim_device, rl_device, graphics_device_id
    )
    return envs
# pyt
config = {
    'seed': 123,
    'task_name': 'Ant',
    'num_envs': 1,
    'sim_device': 'cuda:0',
    'rl_device': 'cuda:0',
    'graphics_device_id': 0
}

env = create_env(
    config['seed'], config['task_name'], config['num_envs'],
    config['sim_device'], config['rl_device'], config['graphics_device_id']
)
model = PPO('MlpPolicy', env, verbose=1)
model.learn(total_timesteps=10000)

obs = env.reset()
for i in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()
    if done:
      obs = env.reset()

env.close()