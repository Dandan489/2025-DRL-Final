import numpy as np
from gym.spaces import Box
from gym_microrts.envs.vec_env import MicroRTSGridModeVecEnv

class MultiDiscrete:
    """
    Action Type: [0, 2] (NOOP, move, attack)
    Move Parameter: [0, 3] (north, east, south, west)
    Relative Attack Position: [0, 48] ((0, 0) to (6, 6))
    """
    def __init__(self, nvec):
        size = len(nvec) // 7
        self.low = np.zeros(3 * size, dtype = int)
        self.high = np.array([2, 3, 48] * size, dtype = int)
        self.num_discrete_space = self.low.shape[0]
        self.n = np.sum(self.high) + 2

    def sample(self):
        """Returns a array with one sample from each discrete action space"""
        # For each row: round(random .* (max - min) + min, 0)
        random_array = np.random.rand(self.num_discrete_space)
        return [int(x) for x in np.floor(np.multiply((self.high - self.low + 1.0), random_array) + self.low)]

    def contains(self, x):
        return (
            len(x) == self.num_discrete_space
            and (np.array(x) >= self.low).all()
            and (np.array(x) <= self.high).all()
        )

    @property
    def shape(self):
        return self.num_discrete_space

    def __repr__(self):
        return "MultiDiscrete" + str(self.num_discrete_space)

    def __eq__(self, other):
        return np.array_equal(self.low, other.low) and np.array_equal(self.high, other.high)


class MicroRTSVecEnv(object):
    def __init__(self, args):
        self.num_features = 18
        self.num_agents = args.num_agents

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
        
        self.observation_space = [self._obs_space_wrapper(self.env.observation_space) for _ in range(self.num_agents)]
        self.share_observation_space = [self._obs_space_wrapper(self.env.observation_space) for _ in range(self.num_agents)]
        self.action_space = [MultiDiscrete(self.env.action_space.nvec) for _ in range(self.num_agents)]

    def _obs_space_wrapper(self, obs_space):
        """
        Hit Points:	5 (0, 1, 2, 3, >=4)
        Owner: 3 (-, player 1, player 2)
        Unit Types:	4 (-, worker, light, heavy, ranged)
        Current Action:	3 (-, move, attack)
        Terrain: 2 (free, wall)
        """
        new_shape = (*obs_space.shape[:-1], self.num_features)
        obs_space = Box(np.zeros(new_shape), np.ones(new_shape), shape = new_shape, dtype = np.int32)
        return obs_space

    def reset(self):
        obs = self.env.reset()
        obs = self._obs_wrapper(obs)
        return obs

    def step(self, action):
        self.env.get_action_mask()
        action = self._action_wrapper(action)
        obs, reward, done, info = self.env.step(action)
        obs = self._obs_wrapper(obs)
        reward = self._reward_wrapper(reward)
        done = self._done_wrapper(done)
        info = self._info_wrapper(info)
        self.env.render()
        return obs, reward, done, info

    def close(self):
        self.env.close()

    def _obs_wrapper(self, obs):
        idx = [0, 1, 2, 3, 4, 10, 11, 12, 13, 18, 19, 20, 21, 22, 26, 27, 28]
        self.agent1 = zip(*np.where(obs[::2, :, :, 11] == 1))
        self.agent2 = zip(*np.where(obs[1::2, :, :, 12] == 1))
        new_obs = np.repeat(obs[:, np.newaxis, :, :, idx], repeats = self.num_agents, axis = 1)
        pos = np.expand_dims(np.zeros(new_obs.shape[:-1]), axis = -1)
        idx1 = [0] * (self.env.num_envs // 2)
        for env, x, y in self.agent1:
            pos[::2][env, idx1[env], x, y, 0] = 1
            idx1[env] += 1
        idx2 = [0] * (self.env.num_envs // 2)
        for env, x, y in self.agent2:
            pos[1::2][env, idx2[env], x, y, 0] = 1
            idx2[env] += 1
        new_obs = np.concatenate((new_obs, pos), axis = -1)
        return new_obs
    
    def _reward_wrapper(self, reward):
        reward = np.expand_dims(np.repeat(reward[:, np.newaxis, ...], self.num_agents, axis = 1), axis = -1)
        return reward
    
    def _done_wrapper(self, done):
        done = np.repeat(done[:, np.newaxis, ...], self.num_agents, axis = 1)
        return done

    def _info_wrapper(self, info):
        return info

    # TODO
    def _action_wrapper(self, action):
        print(action.shape)
        return action
    
    def render(self):
        print('render')