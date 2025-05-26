import numpy as np
from gym_microrts import microrts_ai
from types import SimpleNamespace

from envs.MicroRTS_Env import MicroRTSVecEnv

args = SimpleNamespace()
args.num_selfplay_envs = 8
args.ai2s = []
args.partial_obs = False
args.max_steps = 2000
args.render_theme = 2
args.frame_skip = 0
args.map_paths = ["maps/12x12/basesWorkers12x12.xml"]
args.reward_weight = [0, 1, 0, 0, 0, 5]
args.cycle_maps = []
args.autobuild = False
args.jvm_args = []

env = MicroRTSVecEnv(args)
env.reset()
env.env.get_action_mask()
env.step(np.array([[action_space.sample() for action_space in env.action_space] for _ in range(env.env.num_envs)]))