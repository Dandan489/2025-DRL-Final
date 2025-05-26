import argparse
import os
import random
import subprocess
import time
from distutils.util import strtobool
from typing import List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from gym.spaces import MultiDiscrete
from stable_baselines3.common.vec_env import VecEnvWrapper, VecMonitor, VecVideoRecorder
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter

from gym_microrts import microrts_ai
from gym_microrts.envs.vec_env import MicroRTSGridModeVecEnv

def generate_random_action(h, w, attack_range=7):
    num_tiles = h * w
    action_vector = []

    for _ in range(num_tiles):
        action_type = np.random.randint(0, 6)

        move_param = harvest_param = return_param = produce_dir = 0
        produce_type = 0
        attack_param = 0

        if action_type == 1:
            move_param = np.random.randint(0, 4)
        elif action_type == 2:
            harvest_param = np.random.randint(0, 4)
        elif action_type == 3:
            return_param = np.random.randint(0, 4)
        elif action_type == 4:
            produce_dir = np.random.randint(0, 4)
            produce_type = np.random.randint(0, 7)
        elif action_type == 5:
            attack_param = np.random.randint(0, attack_range ** 2)

        tile_action = [
            action_type,
            move_param,
            harvest_param,
            return_param,
            produce_dir,
            produce_type,
            attack_param
        ]
        action_vector.extend(tile_action)

    return np.array(action_vector, dtype=np.int32)


num_bot_envs = 0
h, w = 24, 24
env = MicroRTSGridModeVecEnv(
            num_selfplay_envs=24,
            num_bot_envs=0,
            partial_obs=False,
            max_steps=2000,
            render_theme=2,
            ai2s=[microrts_ai.coacAI for _ in range(num_bot_envs - 6)]
            + [microrts_ai.randomBiasedAI for _ in range(min(num_bot_envs, 2))]
            + [microrts_ai.lightRushAI for _ in range(min(num_bot_envs, 2))]
            + [microrts_ai.workerRushAI for _ in range(min(num_bot_envs, 2))],
            map_paths=["maps/barricades24x24.xml"],
            reward_weight=np.array([10.0, 1.0, 1.0, 0.2, 1.0, 4.0]),
            cycle_maps="maps/barricades24x24.xml",
        )
obs = env.reset()
done = False
step = 0
while step < (env.max_steps-1) :
    env.get_action_mask()
    actions = np.stack([generate_random_action(h, w) for _ in range(24)], axis=0)
    obs, reward, done, info = env.step(actions)
    step += 1
    env.render()
env.close()
