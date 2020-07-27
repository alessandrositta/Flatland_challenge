import getopt
import random
import sys
from collections import deque
# make sure the root path is in system path
from pathlib import Path

from flatland.envs.malfunction_generators import malfunction_from_params

base_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(base_dir))

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch_training.dueling_double_dqn import Agent

from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import sparse_rail_generator
from flatland.envs.schedule_generators import sparse_schedule_generator
from flatland.utils.rendertools import RenderTool
from utils.observation_utils import normalize_observation
from flatland.envs.observations import TreeObsForRailEnv

def main(argv):
    try:
        opts, args = getopt.getopt(argv, "n:", ["n_trials="])
    except getopt.GetoptError:
        print('training_navigation.py -n <n_trials>')
        sys.exit(2)
    for opt, arg in opts:
        if opt in ('-n', '--n_trials'):
            n_trials = int(arg)

    random.seed(1)
    np.random.seed(1)

    # Parameters for the Environment
    x_dim = 35
    y_dim = 35
    n_agents = 1


    # Use a the malfunction generator to break agents from time to time
    stochastic_data = {'malfunction_rate': 8000,  # Rate of malfunction occurence of single agent
                       'min_duration': 15,  # Minimal duration of malfunction
                       'max_duration': 50  # Max duration of malfunction
                       }

    # Custom observation builder
    TreeObservation = TreeObsForRailEnv(max_depth=2)

    # Different agent types (trains) with different speeds.
    speed_ration_map = {1.: 0.,  # Fast passenger train
                        1. / 2.: 1.0,  # Fast freight train
                        1. / 3.: 0.0,  # Slow commuter train
                        1. / 4.: 0.0}  # Slow freight train

    env = RailEnv(width=x_dim,
                  height=y_dim,
                  rail_generator=sparse_rail_generator(max_num_cities=3,
                                                       # Number of cities in map (where train stations are)
                                                       seed=1,  # Random seed
                                                       grid_mode=False,
                                                       max_rails_between_cities=2,
                                                       max_rails_in_city=3),
                  schedule_generator=sparse_schedule_generator(speed_ration_map),
                  number_of_agents=n_agents,
                  malfunction_generator_and_process_data=malfunction_from_params(stochastic_data),
                  # Malfunction data generator
                  obs_builder_object=TreeObservation)

    # After training we want to render the results so we also load a renderer
    env_renderer = RenderTool(env, gl="PILSVG", )
    # Given the depth of the tree observation and the number of features per node we get the following state_size
    num_features_per_node = env.obs_builder.observation_dim
    tree_depth = 2
    nr_nodes = 0
    for i in range(tree_depth + 1):
        nr_nodes += np.power(4, i)
    state_size = num_features_per_node * nr_nodes

    # The action space of flatland is 5 discrete actions
    action_size = 5

    # We set the number of episodes we would like to train on
    if 'n_trials' not in locals():
        n_trials = 15000

    # And the max number of steps we want to take per episode
    max_steps = int(3 * (env.height + env.width))

    # Define training parameters
    eps = 1.
    eps_end = 0.005
    eps_decay = 0.998

    # And some variables to keep track of the progress
    action_dict = dict()
    final_action_dict = dict()
    scores_window = deque(maxlen=100)
    done_window = deque(maxlen=100)
    scores = []
    dones_list = []
    action_prob = [0] * action_size
    agent_obs = [None] * env.get_num_agents()
    agent_next_obs = [None] * env.get_num_agents()
    agent_obs_buffer = [None] * env.get_num_agents()
    agent_action_buffer = [2] * env.get_num_agents()
    cummulated_reward = np.zeros(env.get_num_agents())
    update_values = False
    # Now we load a Double dueling DQN agent
    agent = Agent(state_size, action_size)

    for trials in range(1, n_trials + 1):

        # Reset environment
        obs, info = env.reset(True, True)
        env_renderer.reset()
        # Build agent specific observations
        for a in range(env.get_num_agents()):
            if obs[a]:
                agent_obs[a] = normalize_observation(obs[a], tree_depth, observation_radius=10)
                agent_obs_buffer[a] = agent_obs[a].copy()

        # Reset score and done
        score = 0
        env_done = 0

        # Run episode
        for step in range(max_steps):
            # Action
            for a in range(env.get_num_agents()):
                if info['action_required'][a]:
                    # If an action is require, we want to store the obs a that step as well as the action
                    update_values = True
                    action = agent.act(agent_obs[a], eps=eps)
                    action_prob[action] += 1
                else:
                    update_values = False
                    action = 0
                action_dict.update({a: action})

            # Environment step
            next_obs, all_rewards, done, info = env.step(action_dict)
            # Update replay buffer and train agent
            for a in range(env.get_num_agents()):
                # Only update the values when we are done or when an action was taken and thus relevant information is present
                if update_values or done[a]:
                    agent.step(agent_obs_buffer[a], agent_action_buffer[a], all_rewards[a],
                               agent_obs[a], done[a])
                    cummulated_reward[a] = 0.

                    agent_obs_buffer[a] = agent_obs[a].copy()
                    agent_action_buffer[a] = action_dict[a]
                if next_obs[a]:
                    agent_obs[a] = normalize_observation(next_obs[a], tree_depth, observation_radius=10)

                score += all_rewards[a] / env.get_num_agents()

            # Copy observation
            if done['__all__']:
                env_done = 1
                break

        # Epsilon decay
        eps = max(eps_end, eps_decay * eps)  # decrease epsilon

        # Collection information about training
        tasks_finished = 0
        for _idx in range(env.get_num_agents()):
            if done[_idx] == 1:
                tasks_finished += 1
        done_window.append(tasks_finished / max(1, env.get_num_agents()))
        scores_window.append(score / max_steps)  # save most recent score
        scores.append(np.mean(scores_window))
        dones_list.append((np.mean(done_window)))

        print(
            '\rTraining {} Agents on ({},{}).\t Episode {}\t Average Score: {:.3f}\tDones: {:.2f}%\tEpsilon: {:.2f} \t Action Probabilities: \t {}'.format(
                env.get_num_agents(), x_dim, y_dim,
                trials,
                np.mean(scores_window),
                100 * np.mean(done_window),
                eps, action_prob / np.sum(action_prob)), end=" ")

        if trials % 100 == 0:
            print(
                '\rTraining {} Agents on ({},{}).\t Episode {}\t Average Score: {:.3f}\tDones: {:.2f}%\tEpsilon: {:.2f} \t Action Probabilities: \t {}'.format(
                    env.get_num_agents(), x_dim, y_dim,
                    trials,
                    np.mean(scores_window),
                    100 * np.mean(done_window),
                    eps, action_prob / np.sum(action_prob)))
            torch.save(agent.qnetwork_local.state_dict(),
                       './Nets/navigator_checkpoint' + str(trials) + '.pth')
            action_prob = [1] * action_size

    # Plot overall training progress at the end
    plt.plot(scores)
    plt.show()


if __name__ == '__main__':
    main(sys.argv[1:])
