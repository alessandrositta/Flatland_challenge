#################################################################################
# Projcect 5 - Flatland challenge                                               #
#                                                                               #
# Test set code - for Results comparison                                        #
#                                                                               #
# @Giovanni Montanari - Lorenzo Sarti - Alessandro Sitta                        #
#                                                                               #
#                                                                               #
# INSTRUCTIONS: To select the test set, please select the parameters at line 52.#
#               Results will be saved in a folder named 'NetsTest'. Please load #
#               your model and its relative weights first(Example at line 178)  #
#################################################################################

import random
import getopt
from collections import deque
import sys
from pathlib import Path
base_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(base_dir))
import numpy as np
import torch
from flatland.envs.malfunction_generators import malfunction_from_params
from flatland.envs.observations import TreeObsForRailEnv
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import sparse_rail_generator
from flatland.envs.schedule_generators import sparse_schedule_generator
from flatland.utils.rendertools import RenderTool, AgentRenderVariant
from importlib_resources import path
from flatland.envs.predictions import ShortestPathPredictorForRailEnv

# Our solution works with dueling double DQN from torch_training (Flatland-baselines). You can import your agent for testing
import torch_training.Nets
from torch_training.dueling_double_dqn import Agent
from utils.observation_utils import normalize_observation

from os import path

def main(argv):
    try:
        opts, args = getopt.getopt(argv, "n:", ["n_trials="])
    except getopt.GetoptError:
        print('test_navigation_single_agent.py -n <n_trials>')
        sys.exit(2)
    for opt, arg in opts:
        if opt in ('-n', '--n_trials'):
            n_trials = int(arg)

    random.seed(1)
    np.random.seed(1)

    ######## TEST SET SELECTION - PARAMETERS ########
    
    test_multi_agent_setup = 1             # 1 for Medium size test, 2 for Big size test
    test_n_agents = 5                      # Number of agents to test (3 - 5 - 7 for Medium, 5 - 7 - 10 for Big)
    test_malfunctions_enabled = True       # Malfunctions enabled?
    test_agents_one_speed = True           # Test agents with the same speed (1) or with 4 different speeds?

    #################################################

    # Medium size
    if test_multi_agent_setup == 1:
        x_dim = 16*3
        y_dim = 9*3
        max_num_cities = 5
        max_rails_between_cities = 2
        max_rails_in_city = 3

    # Big size
    if test_multi_agent_setup == 2:
        x_dim = 16*4
        y_dim = 9*4
        max_num_cities = 9
        max_rails_between_cities = 5
        max_rails_in_city = 5


    stochastic_data = {'malfunction_rate': 80,  # Rate of malfunction occurence of single agent
                       'min_duration': 15,  # Minimal duration of malfunction
                       'max_duration': 50  # Max duration of malfunction
                       }

    # Custom observation builder
    tree_depth = 2
    TreeObservation = TreeObsForRailEnv(max_depth=tree_depth, predictor = ShortestPathPredictorForRailEnv(20))

    np.savetxt(fname=path.join('NetsTest' , 'info.txt'), X=[x_dim,y_dim,test_n_agents,max_num_cities,max_rails_between_cities,max_rails_in_city,tree_depth],delimiter=';')

    # Different agent types (trains) with different speeds.
    if test_agents_one_speed:
        speed_ration_map = {1.: 1.,  # Fast passenger train
                            1. / 2.: 0.0,  # Fast freight train
                            1. / 3.: 0.0,  # Slow commuter train
                            1. / 4.: 0.0}  # Slow freight train
    else:
        speed_ration_map = {1.: 0.25,  # Fast passenger train
                            1. / 2.: 0.25,  # Fast freight train
                            1. / 3.: 0.25,  # Slow commuter train
                            1. / 4.: 0.25}  # Slow freight train

    
    if test_malfunctions_enabled:
        env = RailEnv(width=x_dim,
                      height=y_dim,
                      rail_generator=sparse_rail_generator(max_num_cities=max_num_cities,
                                                           # Number of cities in map (where train stations are)
                                                           seed=14,  # Random seed
                                                           grid_mode=False,
                                                           max_rails_between_cities=max_rails_between_cities,
                                                               max_rails_in_city=max_rails_in_city),
                    schedule_generator=sparse_schedule_generator(speed_ration_map),
                    malfunction_generator_and_process_data=malfunction_from_params(stochastic_data),
                    number_of_agents=test_n_agents,
                    obs_builder_object=TreeObservation)
    else:
        env = RailEnv(width=x_dim,
                      height=y_dim,
                      rail_generator=sparse_rail_generator(max_num_cities=max_num_cities,
                                                           # Number of cities in map (where train stations are)
                                                           seed=14,  # Random seed
                                                           grid_mode=False,
                                                           max_rails_between_cities=max_rails_between_cities,
                                                               max_rails_in_city=max_rails_in_city),
                    schedule_generator=sparse_schedule_generator(speed_ration_map),
                    number_of_agents=test_n_agents,
                    obs_builder_object=TreeObservation)
    
    env.reset()

    #env_renderer = RenderTool(env, gl="PILSVG", )
    env_renderer = RenderTool(env, gl="PILSVG",
                          agent_render_variant=AgentRenderVariant.AGENT_SHOWS_OPTIONS_AND_BOX,
                          show_debug=False,
                          screen_height=(1080*0.8),  # Adjust these parameters to fit your resolution
                          screen_width=(1920*0.8))
    num_features_per_node = env.obs_builder.observation_dim

    
    nr_nodes = 0
    for i in range(tree_depth + 1):
        nr_nodes += np.power(4, i)
    state_size = num_features_per_node * nr_nodes
    action_size = 5

    # We set the number of episodes we would like to train on
    if 'n_trials' not in locals():
        n_trials = 15000
    
    # max_steps computation
    speed_weighted_mean = 0

    for key in speed_ration_map.keys():
        speed_weighted_mean += key * speed_ration_map[key]
    
    #max_steps = int(3 * (env.height + env.width))
    max_steps = int((1/speed_weighted_mean) * 3 * (env.height + env.width))
    #eps = 1.
    #eps_end = 0.005
    #eps_decay = 0.9995

    # And some variables to keep track of the performance
    action_dict = dict()
    final_action_dict = dict()
    action_prob_list = []
    scores_window = deque(maxlen=100)
    done_window = deque(maxlen=100)
    scores = []
    scores_list = []
    deadlock_list =[]
    dones_list_window = []
    dones_list = []
    action_prob = [0] * action_size
    agent_obs = [None] * env.get_num_agents()
    agent_next_obs = [None] * env.get_num_agents() # Useless
    agent = Agent(state_size, action_size)
    
    # LOAD MODEL WEIGHTS TO TEST
    agent.qnetwork_local.load_state_dict(torch.load(path.join('NetsTest' , 'navigator_checkpoint3800_multi10_deadlock_global10.pth')))

    record_images = False
    frame_step = 0

    for trials in range(1, n_trials + 1):

        # Reset environment
        obs, info = env.reset()#(True, True)
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
                    action_prob[action] += 1

                else:
                    action = 0

                action_dict.update({a: action})
            # Environment step
            obs, all_rewards, done, deadlocks, info = env.step(action_dict)

            env_renderer.render_env(show=True, show_predictions=True, show_observations=False)
            # Build agent specific observations and normalize
            for a in range(env.get_num_agents()):
                if obs[a]:
                    agent_obs[a] = normalize_observation(obs[a], tree_depth, observation_radius=10)

                score += all_rewards[a] / env.get_num_agents()


            if done['__all__']:
                break

        # Collection information about training
        tasks_finished = 0
        for _idx in range(env.get_num_agents()):
            if done[_idx] == 1:
                tasks_finished += 1
        done_window.append(tasks_finished / max(1, env.get_num_agents()))
        scores_window.append(score / max_steps)  # save most recent score
        scores.append(np.mean(scores_window))
        dones_list.append(tasks_finished / max(1, env.get_num_agents()))
        dones_list_window.append((np.mean(done_window)))
        scores_list.append(score / max_steps)
        deadlock_list.append(deadlocks.count(1)/max(1, env.get_num_agents()))

        if (np.sum(action_prob) == 0):
            action_prob_normalized = [0] * action_size
        else:
            action_prob_normalized = action_prob / np.sum(action_prob)



        print(
                '\rTesting {} Agents on ({},{}).\t Episode {}\t Score: {:.3f}\tDones: {:.2f}%\tDeadlocks: {:.2f}\t Action Probabilities: \t {}'.format(
                    env.get_num_agents(), x_dim, y_dim,
                    trials,
                    score / max_steps,
                    100 * tasks_finished / max(1, env.get_num_agents()),
                    deadlocks.count(1)/max(1, env.get_num_agents()),
                    action_prob_normalized), end=" ")

        #if trials % 100 == 0:
        action_prob_list.append(action_prob_normalized)
        action_prob = [0] * action_size

        if trials % 50 == 0:

            np.savetxt(fname=path.join('NetsTest' , 'test_metrics.csv'), X=np.transpose(np.asarray([scores_list,scores,dones_list,dones_list_window,deadlock_list])), delimiter=';',newline='\n')
            np.savetxt(fname=path.join('NetsTest' , 'test_action_prob.csv'), X=np.asarray(action_prob_list), delimiter=';',newline='\n')

if __name__ == '__main__':
    main(sys.argv[1:])