import os
import time
import numpy as np
import random
from copy import deepcopy
import torch

from rl_games.common import object_factory
from rl_games.common import tr_helpers

from rl_games.algos_torch import a2c_continuous
from rl_games.algos_torch import a2c_discrete
from rl_games.algos_torch import players
from rl_games.common.algo_observer import DefaultAlgoObserver
from rl_games.algos_torch import sac_agent


def _restore(agent, args):
    if 'checkpoint' in args and args['checkpoint'] is not None and args['checkpoint'] !='':
        agent.restore(args['checkpoint'])

def _override_sigma(agent, args):
    if 'sigma' in args and args['sigma'] is not None:
        net = agent.model.a2c_network
        if hasattr(net, 'sigma') and hasattr(net, 'fixed_sigma'):
            if net.fixed_sigma:
                with torch.no_grad():
                    net.sigma.fill_(float(args['sigma']))
            else:
                print('Print cannot set new sigma because fixed_sigma is False')


class Runner:

    def __init__(self, algo_observer=None):
        print(f"Using Runner from dcd_isaac")
        self.algo_factory = object_factory.ObjectFactory()
        self.algo_factory.register_builder('a2c_continuous', lambda **kwargs : a2c_continuous.A2CAgent(**kwargs))
        self.algo_factory.register_builder('a2c_discrete', lambda **kwargs : a2c_discrete.DiscreteA2CAgent(**kwargs)) 
        self.algo_factory.register_builder('sac', lambda **kwargs: sac_agent.SACAgent(**kwargs))
        #self.algo_factory.register_builder('dqn', lambda **kwargs : dqnagent.DQNAgent(**kwargs))

        self.player_factory = object_factory.ObjectFactory()
        self.player_factory.register_builder('a2c_continuous', lambda **kwargs : players.PpoPlayerContinuous(**kwargs))
        self.player_factory.register_builder('a2c_discrete', lambda **kwargs : players.PpoPlayerDiscrete(**kwargs))
        self.player_factory.register_builder('sac', lambda **kwargs : players.SACPlayer(**kwargs))
        #self.player_factory.register_builder('dqn', lambda **kwargs : players.DQNPlayer(**kwargs))

        self.algo_observer = algo_observer if algo_observer else DefaultAlgoObserver()
        torch.backends.cudnn.benchmark = True
        ### it didnot help for lots for openai gym envs anyway :(
        #torch.backends.cudnn.deterministic = True
        #torch.use_deterministic_algorithms(True)

    def reset(self):
        pass

    def load_config(self, params):
        print(f"Using Runner from dcd_isaac")

        self.seed = params.get('seed', None)
        if self.seed is None:
            self.seed = int(time.time())

        self.local_rank = 0
        self.global_rank = 0
        self.world_size = 1

        if params["config"].get('multi_gpu', False):
            # local rank of the GPU in a node
            self.local_rank = int(os.getenv("LOCAL_RANK", "0"))
            # global rank of the GPU
            self.global_rank = int(os.getenv("RANK", "0"))
            # total number of GPUs across all nodes
            self.world_size = int(os.getenv("WORLD_SIZE", "1"))

            # set different random seed for each GPU
            self.seed += self.global_rank

            print(f"global_rank = {self.global_rank} local_rank = {self.local_rank} world_size = {self.world_size}")

        print(f"self.seed = {self.seed}")

        self.algo_params = params['algo']
        self.algo_name = self.algo_params['name']
        self.exp_config = None

        if self.seed:
            torch.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)
            np.random.seed(self.seed)
            random.seed(self.seed)

            # deal with environment specific seed if applicable
            if 'env_config' in params['config']:
                if not 'seed' in params['config']['env_config']:
                    params['config']['env_config']['seed'] = self.seed
                else:
                    if params["config"].get('multi_gpu', False):
                        params['config']['env_config']['seed'] += self

        config = params['config']
        config['reward_shaper'] = tr_helpers.DefaultRewardsShaper(**config['reward_shaper'])
        if 'features' not in config:
            config['features'] = {}
        config['features']['observer'] = self.algo_observer
        self.params = params

    def load(self, yaml_config):
        config = deepcopy(yaml_config)
        self.default_config = deepcopy(config['params'])
        self.load_config(params=self.default_config)

    def run_train(self, args):
        print('Started to train')
        agent = self.algo_factory.create(self.algo_name, base_name='run', params=self.params)
        _restore(agent, args)
        _override_sigma(agent, args)
        agent.train()

    def run_play(self, args):
        print('Started to play')
        player = self.create_player()
        _restore(player, args)
        _override_sigma(player, args)
        player.run()

    def create_player(self):
        return self.player_factory.create(self.algo_name, params=self.params)

    def reset(self):
        pass

    def run(self, args):
        if args['train']:
            if 'second_agent_args' in args:
                self.run_two_agents_train(args, args['second_agent_args'])
            else:
                self.run_train(args)
        elif args['play']:
            self.run_play(args)
        else:
            self.run_train(args)

    def run_two_agents_train(self, args1, args2):
        print('Started to train two agents')

        # Initialize two agents
        agent1 = self.algo_factory.create(self.algo_name, base_name='run1', params=args1)
        agent2 = self.algo_factory.create(self.algo_name, base_name='run2', params=args2)

        # Restore from checkpoints if provided
        _restore(agent1, args1)
        _restore(agent2, args2)

        # Override sigma if provided
        _override_sigma(agent1, args1)
        _override_sigma(agent2, args2)

        # Synchronized training loop
        for _ in range(self.algo_params['num_epochs']):
            agent1.train_step()
            agent2.train_step()

            # Synchronize (pseudo-code, actual synchronization depends on your setup)
            self.synchronize_agents(agent1, agent2)
    def synchronize_agents(self, agent1, agent2):
        # Placeholder for synchronization logic
        # Depending on your setup (e.g., if you're using multi-threading or a distributed setup),
        # you'll need to implement this function to ensure both agents have finished their updates
        # before proceeding.
        pass

# class Runner:
#     # ... other methods ...

#     def run_two_agents_train(self, args1, args2):
#         print('Started to train two agents')

#         # Modify the configuration for each agent to specify the environment
#         params1 = deepcopy(self.params)
#         params1['env_info'] = 'envs1_specific_info'  # Replace with actual info or reference to envs1
#         params2 = deepcopy(self.params)
#         params2['env_info'] = 'envs2_specific_info'  # Replace with actual info or reference to envs2

#         # Initialize two agents with their respective environments
#         agent1 = self.algo_factory.create(self.algo_name, base_name='run1', params=params1)
#         agent2 = self.algo_factory.create(self.algo_name, base_name='run2', params=params2)

#         # ... rest of the method ...
