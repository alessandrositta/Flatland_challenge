import random
from collections import deque

import numpy as np
import torch
from flatland.envs.malfunction_generators import malfunction_from_params
from flatland.envs.observations import TreeObsForRailEnv
from flatland.envs.predictions import ShortestPathPredictorForRailEnv
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import sparse_rail_generator
from flatland.envs.schedule_generators import sparse_schedule_generator
from flatland.utils.rendertools import RenderTool
from importlib_resources import path

import torch_training.Nets
from torch_training.dueling_double_dqn import Agent
from utils.observation_utils import normalize_observation

random.seed(1)
np.random.seed(1)
"""
file_name = "./railway/complex_scene.pkl"
env = RailEnv(width=10,
              height=20,
              rail_generator=rail_from_file(file_name),
              obs_builder_object=TreeObsForRailEnv(max_depth=3, predictor=ShortestPathPredictorForRailEnv()))
x_dim = env.width
y_dim = env.height
"""

# Parameters for the Environment
x_dim = 25
y_dim = 25
n_agents = 10

# We are training an Agent using the Tree Observation with depth 2
observation_builder = TreeObsForRailEnv(max_depth=2)

# Use a the malfunction generator to break agents from time to time
stochastic_data = {'malfunction_rate': 8000,  # Rate of malfunction occurence of single agent
                   'min_duration': 15,  # Minimal duration of malfunction
                   'max_duration': 50  # Max duration of malfunction
                   }


# Custom observation builder
TreeObservation = TreeObsForRailEnv(max_depth=2, predictor=ShortestPathPredictorForRailEnv(30))

# Different agent types (trains) with different speeds.
speed_ration_map = {1.: 0.25,  # Fast passenger train
                    1. / 2.: 0.25,  # Fast freight train
                    1. / 3.: 0.25,  # Slow commuter train
                    1. / 4.: 0.25}  # Slow freight train

env = RailEnv(width=x_dim,
              height=y_dim,
              rail_generator=sparse_rail_generator(max_num_cities=3,
                                                   # Number of cities in map (where train stations are)
                                                   seed=1,  # Random seed
                                                   grid_mode=False,
                                                   max_rails_between_cities=2,
                                                   max_rails_in_city=2),
              schedule_generator=sparse_schedule_generator(speed_ration_map),
              number_of_agents=n_agents,
              malfunction_generator_and_process_data=malfunction_from_params(stochastic_data),
              obs_builder_object=TreeObservation)
env.reset(True, True)

observation_helper = TreeObsForRailEnv(max_depth=3, predictor=ShortestPathPredictorForRailEnv())
env_renderer = RenderTool(env, gl="PILSVG", )
num_features_per_node = env.obs_builder.observation_dim

tree_depth = 2
nr_nodes = 0
for i in range(tree_depth + 1):
    nr_nodes += np.power(4, i)
state_size = num_features_per_node * nr_nodes
action_size = 5

# We set the number of episodes we would like to train on
if 'n_trials' not in locals():
    n_trials = 60000
max_steps = int(4 * 2 * (20 + env.height + env.width))
eps = 1.
eps_end = 0.005
eps_decay = 0.9995
action_dict = dict()
final_action_dict = dict()
scores_window = deque(maxlen=100)
done_window = deque(maxlen=100)
scores = []
dones_list = []
action_prob = [0] * action_size
agent_obs = [None] * env.get_num_agents()
agent_next_obs = [None] * env.get_num_agents()
agent = Agent(state_size, action_size)
with path(torch_training.Nets, "navigator_checkpoint1200.pth") as file_in:
    agent.qnetwork_local.load_state_dict(torch.load(file_in))

record_images = False
frame_step = 0

for trials in range(1, n_trials + 1):

    # Reset environment
    obs, info = env.reset(True, True)
    env_renderer.reset()
    # Build agent specific observations
    for a in range(env.get_num_agents()):
        agent_obs[a] = agent_obs[a] = normalize_observation(obs[a], tree_depth, observation_radius=10)
    # Reset score and done
    score = 0
    env_done = 0

    # Run episode
    for step in range(max_steps):

        # Action
        for a in range(env.get_num_agents()):
            if info['action_required'][a]:
                action = agent.act(agent_obs[a], eps=0.)
            else:
                action = 0

            action_prob[action] += 1
            action_dict.update({a: action})
        # Environment step
        obs, all_rewards, done, _ = env.step(action_dict)
        env_renderer.render_env(show=True, show_predictions=True, show_observations=False)
        # Build agent specific observations and normalize
        for a in range(env.get_num_agents()):
            if obs[a]:
                agent_obs[a] = normalize_observation(obs[a], tree_depth, observation_radius=10)


        if done['__all__']:
            break

