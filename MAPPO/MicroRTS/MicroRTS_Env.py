from gym import spaces
from gym_microrts.envs.vec_env import MicroRTSGridModeVecEnv
import numpy as np
import random


class MicroRTSVecEnv(object):
    def __init__(self, args):
        self.num_agents = 10 # TODO

        self.env = MicroRTSGridModeVecEnv(
            num_selfplay_envs = args.num_selfplay_envs,
            num_bot_envs = len(args.ai2s),
            partial_obs = args.partial_obs,
            max_steps = args.max_steps,
            render_theme = args.render_theme,
            frame_skip = args.frame_skip,
            ai2s = args.ai2s,
            map_paths = args.map_paths,
            reward_weight = args.reward_weight,
            cycle_maps = args.cycle_maps,
            autobuild = args.autobuild,
            jvm_args = args.jvm_args
        )
        
        self.action_space = [self.env.action_space for _ in range(self.num_agents)]
        self.observation_space = [self.env.observation_space for _ in range(self.num_agents)]

    def reset(self):
        obs = self.env.reset()
        obs = self._obs_wrapper(obs)
        return obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        obs = self._obs_wrapper(obs)
        reward = self._reward_wrapper(reward)
        done = self._done_wrapper(done)
        info = self._info_wrapper(info)
        return obs, reward, done, info

    def seed(self, seed=None):
        if seed is None:
            random.seed(1)
        else:
            random.seed(seed)

    def close(self):
        self.env.close()

    # TODO
    def _obs_wrapper(self, obs):
        return obs
    
    # TODO
    def _reward_wrapper(self, reward):
        return reward
    
    # TODO
    def _done_wrapper(self, done):
        return done

    # TODO
    def _info_wrapper(self, info):
        return info
