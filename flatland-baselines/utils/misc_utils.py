import random
import time
from collections import deque

import numpy as np
from flatland.envs.observations import GlobalObsForRailEnv
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import complex_rail_generator
from flatland.envs.schedule_generators import complex_schedule_generator
from line_profiler import LineProfiler

from utils.observation_utils import norm_obs_clip, split_tree_into_feature_groups


def printProgressBar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='*'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '_' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end=" ")
    # Print New Line on Complete
    if iteration == total:
        print('')


class RandomAgent:

    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size

    def act(self, state, eps=0):
        """
        :param state: input is the observation of the agent
        :return: returns an action
        """
        return np.random.choice(np.arange(self.action_size))

    def step(self, memories):
        """
        Step function to improve agent by adjusting policy given the observations

        :param memories: SARS Tuple to be
        :return:
        """
        return

    def save(self, filename):
        # Store the current policy
        return

    def load(self, filename):
        # Load a policy
        return


def run_test(parameters, agent, test_nr=0, tree_depth=3):
    # Parameter initialization
    lp = LineProfiler()
    features_per_node = 9
    start_time_scoring = time.time()
    action_dict = dict()
    nr_trials_per_test = 5
    print('Running Test {} with (x_dim,y_dim) = ({},{}) and {} Agents.'.format(test_nr, parameters[0], parameters[1],
                                                                               parameters[2]))

    # Reset all measurements
    time_obs = deque(maxlen=2)
    test_scores = []
    test_dones = []

    # Reset environment
    random.seed(parameters[3])
    np.random.seed(parameters[3])
    nr_paths = max(2, parameters[2] + int(0.5 * parameters[2]))
    min_dist = int(min([parameters[0], parameters[1]]) * 0.75)
    env = RailEnv(width=parameters[0],
                  height=parameters[1],
                  rail_generator=complex_rail_generator(nr_start_goal=nr_paths, nr_extra=5, min_dist=min_dist,
                                                        max_dist=99999,
                                                        seed=parameters[3]),
                  schedule_generator=complex_schedule_generator(),
                  obs_builder_object=GlobalObsForRailEnv(),
                  number_of_agents=parameters[2])
    max_steps = int(3 * (env.height + env.width))
    lp_step = lp(env.step)
    lp_reset = lp(env.reset)

    agent_obs = [None] * env.get_num_agents()
    printProgressBar(0, nr_trials_per_test, prefix='Progress:', suffix='Complete', length=20)
    for trial in range(nr_trials_per_test):
        # Reset the env

        lp_reset(True, True)
        obs, info = env.reset(True, True)
        for a in range(env.get_num_agents()):
            data, distance, agent_data = split_tree_into_feature_groups(obs[a], tree_depth)
            data = norm_obs_clip(data)
            distance = norm_obs_clip(distance)
            agent_data = np.clip(agent_data, -1, 1)
            obs[a] = np.concatenate((np.concatenate((data, distance)), agent_data))

        for i in range(2):
            time_obs.append(obs)

        for a in range(env.get_num_agents()):
            agent_obs[a] = np.concatenate((time_obs[0][a], time_obs[1][a]))

        # Run episode
        trial_score = 0
        for step in range(max_steps):

            for a in range(env.get_num_agents()):
                action = agent.act(agent_obs[a], eps=0)
                action_dict.update({a: action})

            # Environment step
            next_obs, all_rewards, done, _ = lp_step(action_dict)

            for a in range(env.get_num_agents()):
                data, distance, agent_data = split_tree_into_feature_groups(next_obs[a], tree_depth)
                data = norm_obs_clip(data)
                distance = norm_obs_clip(distance)
                agent_data = np.clip(agent_data, -1, 1)
                next_obs[a] = np.concatenate((np.concatenate((data, distance)), agent_data))
            time_obs.append(next_obs)
            for a in range(env.get_num_agents()):
                agent_obs[a] = np.concatenate((time_obs[0][a], time_obs[1][a]))
                trial_score += all_rewards[a] / env.get_num_agents()

            if done['__all__']:
                break
        test_scores.append(trial_score / max_steps)
        test_dones.append(done['__all__'])
        printProgressBar(trial + 1, nr_trials_per_test, prefix='Progress:', suffix='Complete', length=20)
    end_time_scoring = time.time()
    tot_test_time = end_time_scoring - start_time_scoring
    lp.print_stats()
    return test_scores, test_dones, tot_test_time
