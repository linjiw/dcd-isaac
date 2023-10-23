# Configuration Summary

## Environment
- **Environment Name**: MultiGrid-GoalLastFewerBlocksAdversarial-v0

## Algorithm
- **UED Algorithm**: PAIRED
- **Processes**: 16
- **Environment Steps**: 2500
- **Steps**: 256
- **PPO Epoch**: 5
- **Mini Batches**: 1
- **Handle Time Limits**: True

## Learning
- **Learning Rate**: 1e-4
- **Gamma**: 0.995
- **Entropy Coefficient**: 0.0
- **Adversary Entropy Coefficient**: 0.0

## Recurrent Settings
- **Architecture**: LSTM
- **Recurrent Agent**: True
- **Recurrent Adversary Environment**: True
- **Hidden Size**: 256

## Logging
- **Log Directory**: ~/logs/dcd
- **Log Interval**: 25
- **Log Action Complexity**: True
- **Archive Interval**: 30518
- **Log PLR Buffer Stats**: True
- **Log Replay Complexity**: True
- **Reject Unsolvable Seeds**: False

## Checkpointing
- **Checkpoint**: True

# Training Logic Breakdown

## 1. Environment Setup
- Create parallel environments using `create_parallel_env`.
- Determine if the environment is trainable and if the algorithm is paired.
- Create the main agent using `make_agent`.
- If the algorithm is paired, create the adversary agent.
- If the environment is trainable, create the adversary environment.
- If using domain randomization and level replay, initialize the adversary environment.

## 2. Runner Setup
- If using Protagonist Level Replay (PLR), prepare the PLR arguments.
- Create an `AdversarialRunner` to manage the training loop.

## 3. Checkpoint Configuration
- Set up paths for checkpointing.
- Define a `checkpoint` function to save the model's state.
  
## 4. Load Checkpoint (if available)
- If a checkpoint exists, load it to resume training.
- If fine-tuning, load the model from a specified checkpoint.

## 5. Evaluator Setup
- If test environments are specified, set up an evaluator to assess the agent's performance.

## 6. Main Training Loop
- Determine the total number of updates.
- For each update:
  - Run the training iteration using `train_runner.run()`.
  - Log statistics at specified intervals.
  - Evaluate the agent's performance if required.
  - Save checkpoints at specified intervals.
  - If screenshot interval is set, save screenshots of the environment.

## 7. Cleanup
- Close the evaluator and the environment.


# AdversarialRunner Class Breakdown

The `AdversarialRunner` class manages the rollouts of an adversarial environment. It works with a protagonist (agent), antagonist (adversary_agent), and environment adversary (adversary_env).

## 1. Initialization (`__init__`)

- **Arguments**:
  - `venv`: Vectorized, adversarial gym environment with agent-specific wrappers.
  - `agent`: Protagonist trainer.
  - `ued_venv`: Vectorized, adversarial gym environment with adversary-env-specific wrappers.
  - `adversary_agent`: Antagonist trainer.
  - `adversary_env`: Environment adversary trainer.
  - `flexible_protagonist`: Determines which agent plays the role of protagonist based on the lowest score.

- **Environment Setup**:
  - Determine if the environment has discrete actions.
  - Create a dictionary of agents: protagonist, antagonist, and environment adversary.

- **Rollout Steps Configuration**:
  - Set the number of rollout steps for the agent and the adversary environment.

- **Algorithm Type Checks**:
  - Check if the algorithm is of type domain randomization, paired, or minimax.

- **Training Mode**:
  - If in training mode, call the `train()` method, otherwise call the `eval()` method.

- **Reset**:
  - Reset the runner's state.

- **Protagonist Level Replay (PLR) Setup**:
  - If using PLR, set up the level store, level samplers, and other related configurations.

- **ALP-GMM Setup**:
  - If using the ALP-GMM algorithm, initialize its specific configurations.

## 2. Properties

- **use_byte_encoding**:
  - Determines if byte encoding should be used based on the environment name and other configurations.

## 3. ALP-GMM Initialization (`_init_alp_gmm`)

- Set up the ALP-GMM teacher based on the environment type and its bounds.


# model_for_multigrid_agent Function Breakdown

The `model_for_multigrid_agent` function constructs a model for a given agent type in a MultiGrid environment.

## 1. Function Arguments

- `env`: The environment instance.
- `agent_type`: The type of agent (default is 'agent').
- `recurrent_arch`: The architecture type for recurrent layers.
- `recurrent_hidden_size`: The hidden size for recurrent layers.
- `use_global_critic`: A flag to determine if a global critic should be used.
- `use_global_policy`: A flag to determine if a global policy should be used.

## 2. Adversary Environment Agent

If the `agent_type` is 'adversary_env':
- Extract the observation and action spaces specific to the adversary environment.
- Determine the maximum timestep and random_z dimension from the observation space.
- Construct the `MultigridNetwork` model with the extracted parameters and other configurations.

## 3. Other Agents

For agent types other than 'adversary_env':
- Extract the general observation and action spaces.
- Determine the number of directions from the observation space.
- Set up the model arguments.
- Choose the model constructor based on the `use_global_critic` flag.
- Update the model arguments if `use_global_policy` is True.
- Construct the model using the chosen constructor and arguments.

## 4. Return

The function returns the constructed model.

## Understanding Regret Calculation in Adversarial Training

The code provided is part of an adversarial training setup, where an agent is trained against an adversary. The regret is a measure of how well the agent performs against the adversary. Let's break down the code to understand how regret is calculated and used:

### `_compute_env_return` Function:

This function computes the regret based on the performance of the agent and the adversary agent.

#### Parameters:
- `agent_info`: Information about the agent's performance.
- `adversary_agent_info`: Information about the adversary agent's performance.

#### Logic:
1. If the algorithm used is 'paired':
    - The regret (`env_return`) is the difference between the maximum return of the adversary agent and the mean return of the agent. However, it's capped at a minimum of zero using `torch.max`.
    
2. If the algorithm used is 'flexible_paired':
    - Initialize a tensor `env_return` with zeros.
    - Determine which returns (agent's or adversary's) are greater.
    - For the indices where the adversary's return is greater, set `env_return` to the adversary's max return. For the other indices, set it to the agent's max return.
    - Similarly, compute the mean return for the environment (`env_mean_return`) based on which agent's return is greater.
    - Finally, the regret (`env_return`) is the difference between `env_return` and `env_mean_return`, capped at a minimum of zero.

### `run` Function:

This function orchestrates the training loop, where the agent and the adversary agent interact with the environment.

#### Key Points:
1. The function starts by setting up the agents and determining if the current iteration will involve replaying levels (`level_replay`).
2. A batch of adversarial environments is generated.
3. The agent and the adversary agent interact with these environments, and their performance metrics are stored in `agent_info` and `adversary_agent_info`, respectively.
4. If the setup uses the ACCEL method, levels are edited and evaluated.
5. The regret is computed using the `_compute_env_return` function with the `agent_info` and `adversary_agent_info` as inputs.
6. If the environment is being trained, the regret is used to update the adversary environment's returns.
7. Logging is done to capture various statistics, including the regret.

#### How to get the parameters:
- `agent_info` and `adversary_agent_info` are obtained from the `agent_rollout` function, which likely simulates the agent's and adversary agent's interactions with the environment and returns performance metrics.

In summary, the regret is a measure of the difference between the agent's performance and the adversary agent's performance. In the 'paired' setup, it's the difference between the adversary's best performance and the agent's average performance. In the 'flexible_paired' setup, it's more dynamic, considering the better performer between the agent and the adversary for each instance. This regret is then used to update the adversary environment, guiding its training to generate more challenging scenarios for the agent.

# How to translate from dcd to IsaacGym

- Use adversarial runner (./envs/runner/adversarial_runner.py)
- check how agents run train() (agent = ACAgent(algo=algo, storage=storage).to(device))
- how trains not matter, where did the reward change?



./dcd-isaac
├── algos
│   ├── agent.py
│   ├── __init__.py
│   ├── ppo.py
│   └── storage.py
├── arguments.py
├── docs
│   ├── bibtex
│   │   ├── accel.bib
│   │   └── dcd.bib
│   ├── CODE_OF_CONDUCT.md
│   ├── CONTRIBUTING.md
│   └── images
│       ├── accel_bipedal_demo.gif
│       ├── bipedal_challenges.png
│       ├── car_racing_bezier_tracks.png
│       ├── dcd_overview_darkmode.png
│       ├── dcd_overview.png
│       ├── f1_benchmark_tracks.png
│       └── ood_mazes.png
├── envs
│   ├── bipedalwalker
│   │   ├── adversarial.py
│   │   ├── __init__.py
│   │   ├── register.py
│   │   ├── walker_env.py
│   │   └── walker_test_envs.py
│   ├── box2d
│   │   ├── bezier.py
│   │   ├── car_racing_adversarial.py
│   │   ├── car_racing_bezier.py
│   │   ├── car_racing_f1.py
│   │   ├── car_racing.py
│   │   ├── __init__.py
│   │   ├── racetracks
│   │   │   ├── formula1.py
│   │   │   ├── __init__.py
│   │   │   └── racetrack.py
│   │   └── rendering.py
│   ├── __init__.py
│   ├── multigrid
│   │   ├── adversarial.py
│   │   ├── cluttered.py
│   │   ├── crossing.py
│   │   ├── fourrooms.py
│   │   ├── __init__.py
│   │   ├── manual_control.py
│   │   ├── maze.py
│   │   ├── mst_maze.py
│   │   ├── multigrid.py
│   │   ├── register.py
│   │   └── window.py
│   ├── registration.py
│   ├── runners
│   │   ├── adversarial_runner.py
│   │   └── __init__.py
│   └── wrappers
│       ├── car_racing_wrappers.py
│       ├── __init__.py
│       ├── multigrid_wrappers.py
│       ├── obs_wrappers.py
│       ├── parallel_wrappers.py
│       ├── time_limit.py
│       ├── vec_env.py
│       ├── vec_frame_stack.py
│       ├── vec_monitor.py
│       └── vec_normalize.py
├── eval.py
├── level_replay
│   ├── __init__.py
│   ├── level_sampler.py
│   └── level_store.py
├── LICENSE
├── models
│   ├── car_racing_models.py
│   ├── common.py
│   ├── distributions.py
│   ├── __init__.py
│   ├── multigrid_global_critic_models.py
│   ├── multigrid_models.py
│   ├── popart.py
│   └── walker_models.py
├── notes.md
├── README.MD
├── requirements.txt
├── results
│   ├── bipedal
│   │   ├── bipedal8d-accel_20k-updates.csv
│   │   ├── bipedal8d-dr_20k-updates.csv
│   │   └── bipedal8d-robust_plr-20k_updates.csv
│   ├── car_racing_f1
│   │   ├── f1-dr-5M_steps.csv
│   │   ├── f1-paired-5M_steps.csv
│   │   ├── f1-plr-5M_steps.csv
│   │   ├── f1-repaired-5M_steps.csv
│   │   └── f1-robust_plr-5M_steps.csv
│   ├── minigrid_ood
│   │   ├── mg_25_blocks-dr-250M_steps.csv
│   │   ├── mg_25_blocks-minimax-250M_steps.csv
│   │   ├── mg_25_blocks-paired-250M_steps.csv
│   │   ├── mg_25_blocks-plr-250M_steps.csv
│   │   ├── mg_25_blocks-repaired-250M_steps.csv
│   │   ├── mg_25_blocks-robust_plr-250M_steps.csv
│   │   ├── mg_60_blocks-accel_20k_updates.csv
│   │   ├── mg_60_blocks_uni-dr_20k_updates.csv
│   │   └── mg_60_blocks_uni-robust_plr_20k_updates.csv
│   ├── plot_eval_bars.py
│   └── plot_f1.py
├── teachDeepRL
│   ├── __init__.py
│   └── teachers
│       ├── algos
│       │   ├── alp_gmm.py
│       │   ├── covar_gmm.py
│       │   ├── __init__.py
│       │   ├── oracle_teacher.py
│       │   ├── random_teacher.py
│       │   └── riac.py
│       ├── __init__.py
│       ├── teacher_controller.py
│       └── utils
│           ├── dataset.py
│           ├── plot_utils.py
│           └── test_utils.py
├── train.py
├── train_scripts
│   ├── grid_configs
│   │   ├── bipedal
│   │   │   ├── bipedal_accel.json
│   │   │   ├── bipedal_accel_poet.json
│   │   │   ├── bipedal_alpgmm.json
│   │   │   ├── bipedal_dr.json
│   │   │   ├── bipedal_minimax.json
│   │   │   ├── bipedal_paired.json
│   │   │   └── bipedal_robust_plr.json
│   │   ├── car_racing
│   │   │   ├── cr_dr.json
│   │   │   ├── cr_paired.json
│   │   │   ├── cr_plr.json
│   │   │   ├── cr_repaired.json
│   │   │   └── cr_robust_plr.json
│   │   └── minigrid
│   │       ├── 25_blocks
│   │       │   ├── mg_25b_dr.json
│   │       │   ├── mg_25b_minimax.json
│   │       │   ├── mg_25b_paired.json
│   │       │   ├── mg_25b_plr.json
│   │       │   ├── mg_25b_repaired.json
│   │       │   └── mg_25b_robust_plr.json
│   │       └── 60_blocks_uniform
│   │           ├── mg_60b_uni_accel_empty.json
│   │           ├── mg_60b_uni_dr.json
│   │           └── mg_60b_uni_robust_plr.json
│   └── make_cmd.py
└── util
    ├── filewriter.py
    ├── geo_complexity.py
    ├── __init__.py
    ├── make_agent.py
    └── unionfind.py


tree ./isaacgym-jackal -I 'image_folder|generated_npy|generated_images|*.jpg|*.png|*.pyc|run-*|*.pth' -L 2

./isaacgym-jackal
├── assets
│   ├── amp
│   ├── factory
│   ├── glb
│   ├── licenses
│   ├── mjcf
│   ├── trifinger
│   └── urdf
├── docs
│   ├── domain_randomization.md
│   ├── factory.md
│   ├── framework.md
│   ├── images
│   ├── release_notes.md
│   ├── reproducibility.md
│   └── rl_examples.md
├── gan.py
├── isaacgymenvs
│   ├── cfg
│   ├── gan_checkpoint
│   ├── gan.py
│   ├── GAN.py
│   ├── generate_maze.py
│   ├── __init__.py
│   ├── jackal.md
│   ├── learning
│   ├── maze
│   ├── PAIR.py
│   ├── __pycache__
│   ├── runs
│   ├── saved_data.npz
│   ├── simple_snippet.py
│   ├── tasks
│   ├── test_bo.py
│   ├── test_import.py
│   ├── test.md
│   ├── test_plot.py
│   ├── test.py
│   ├── train.py
│   ├── train_two.py
│   ├── utils
│   ├── videos
│   └── wandb
├── isaacgymenvs.egg-info
│   ├── dependency_links.txt
│   ├── not-zip-safe
│   ├── PKG-INFO
│   ├── requires.txt
│   ├── SOURCES.txt
│   └── top_level.txt
├── LICENSE.txt
├── python
│   └── example
├── README.md
├── rl_games
│   ├── docs
│   ├── LICENSE
│   ├── notebooks
│   ├── poetry.lock
│   ├── pyproject.toml
│   ├── README.md
│   ├── rl_games
│   ├── runner.py
│   ├── setup.py
│   └── tests
└── setup.py