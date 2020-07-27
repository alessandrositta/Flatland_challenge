import numpy as np
from flatland.envs.observations import TreeObsForRailEnv
from flatland.envs.predictions import ShortestPathPredictorForRailEnv
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import complex_rail_generator
from flatland.envs.schedule_generators import complex_schedule_generator
from flatland.utils.rendertools import RenderTool

from sequential_agent.simple_order_agent import OrderedAgent

np.random.seed(2)
"""
file_name = "../torch_training/railway/complex_scene.pkl"
env = RailEnv(width=10,
              height=20,
              rail_generator=rail_from_file(file_name),
              obs_builder_object=TreeObsForRailEnv(max_depth=1, predictor=ShortestPathPredictorForRailEnv()))
x_dim = env.width
y_dim = env.height

"""

x_dim = 20  # np.random.randint(8, 20)
y_dim = 20  # np.random.randint(8, 20)
n_agents = 10  # np.random.randint(3, 8)
n_goals = n_agents + np.random.randint(0, 3)
min_dist = int(0.75 * min(x_dim, y_dim))

env = RailEnv(width=x_dim,
              height=y_dim,
              rail_generator=complex_rail_generator(nr_start_goal=n_goals, nr_extra=5, min_dist=min_dist,
                                                    max_dist=99999,
                                                    seed=0),
              schedule_generator=complex_schedule_generator(),
              obs_builder_object=TreeObsForRailEnv(max_depth=1, predictor=ShortestPathPredictorForRailEnv()),
              number_of_agents=n_agents)
env.reset(True, True)

tree_depth = 1
observation_helper = TreeObsForRailEnv(max_depth=tree_depth, predictor=ShortestPathPredictorForRailEnv())
env_renderer = RenderTool(env, gl="PILSVG", )
handle = env.get_agent_handles()
n_trials = 1
max_steps = 100 * (env.height + env.width)
record_images = False
agent = OrderedAgent()
action_dict = dict()

for trials in range(1, n_trials + 1):

    # Reset environment
    obs, info = env.reset(True, True)
    done = env.dones
    env_renderer.reset()
    frame_step = 0
    # Run episode
    for step in range(max_steps):
        env_renderer.render_env(show=True, show_observations=False, show_predictions=True)

        if record_images:
            env_renderer.gl.save_image("./Images/flatland_frame_{:04d}.bmp".format(frame_step))
            frame_step += 1

        # Action
        acting_agent = 0
        for a in range(env.get_num_agents()):
            if done[a]:
                acting_agent += 1
            if a == acting_agent:
                action = agent.act(obs[a], eps=0)
                print(action)
            else:
                action = 4
            action_dict.update({a: action})

        # Environment step

        obs, all_rewards, done, _ = env.step(action_dict)

        if done['__all__']:
            break
