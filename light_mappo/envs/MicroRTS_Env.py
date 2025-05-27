import numpy as np
from gym.spaces import Box
from gym_microrts.envs.vec_env import MicroRTSGridModeVecEnv

class MultiDiscrete:
    """
    Action Type: [0, 2] (NOOP, move, attack)
    Move Parameter: [0, 3] (north, east, south, west)
    Relative Attack Position: [0, 48] ((0, 0) to (6, 6))
    """
    def __init__(self):
        self.low = np.zeros(3, dtype = int)
        self.high = np.array([2, 3, 48], dtype = int)
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
        self.action_space = [MultiDiscrete() for _ in range(self.num_agents)]
        
        self.agent_id_pos_map = [None] * self.env.num_envs
        self.to_reset = [False] * self.env.num_envs

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
        self.env.get_action_mask()
        for env_idx in range(self.env.num_envs):
            self._reset_agent_positions(env_idx, obs)
        obs = self._obs_wrapper(obs)
        return obs

    def step(self, action):
        action = self._action_wrapper(action)
        obs, reward, done, info = self.env.step(action)
        for env_idx, to_reset in enumerate(self.to_reset):
            if to_reset:
                self._reset_agent_positions(env_idx, obs)
        self._update_agent_positions(action, obs)
        for env_idx in range(self.env.num_envs):
            if done[env_idx]:
                self.to_reset[env_idx] = True
        self.env.get_action_mask()
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
        self.agent = list(zip(*np.where(obs[..., 11] == 1)))

        # if agent doesn't exist from obs, remove it from agent_id_pos_map
        new_agent_id_pos_map = []
        for env_idx in range(self.env.num_envs):
            env_dict = {}
            for agent_id, (x, y, t, nx, ny) in self.agent_id_pos_map[env_idx].items():
                if obs[env_idx, x, y, 11] == 1:
                    env_dict[agent_id] = (x, y, t, nx, ny)
            new_agent_id_pos_map.append(env_dict)
        self.agent_id_pos_map = new_agent_id_pos_map
            
        new_obs = np.repeat(obs[:, np.newaxis, :, :, idx], repeats=self.num_agents, axis=1)
        pos = np.expand_dims(np.zeros(new_obs.shape[:-1]), axis=-1)

        for env_idx in range(self.env.num_envs):
            for agent_id, (x, y, *_) in self.agent_id_pos_map[env_idx].items():
                pos[env_idx, agent_id, x, y, 0] = 1

        new_obs = np.concatenate((new_obs, pos), axis=-1)
        return new_obs
    
    def _reward_wrapper(self, reward):
        reward = np.expand_dims(np.repeat(reward[:, np.newaxis, ...], self.num_agents, axis = 1), axis = -1)
        return reward
    
    def _done_wrapper(self, done):
        done = np.repeat(done[:, np.newaxis, ...], self.num_agents, axis = 1)
        return done

    def _info_wrapper(self, info):
        return info

    def _action_wrapper(self, action):
        new_action = np.zeros((self.env.num_envs, self.env.height, self.env.width, 7))
        action = np.where(action == 1)[-1].reshape((self.env.num_envs, self.num_agents, -1))
        action -= np.array([0, 3, 7])
        for env_idx in range(self.env.num_envs):
            for agent_id, (x, y, *_) in self.agent_id_pos_map[env_idx].items():
                new_action[env_idx, x, y, 0] = 5 if action[env_idx, agent_id, 0] == 2 else action[env_idx, agent_id, 0]
                new_action[env_idx, x, y, [1, 6]] = action[env_idx, agent_id, [1, 2]]

        return new_action
    
    def render(self):
        print('render')


    def _reset_agent_positions(self, env_idx, obs):
        """
        for example self.agent_id_pos_map = [
            # Env 0  {agent_id: (agent_pos, t_to_move, next_pos), ...}
            { 0: (1, 3, -1, 0, 0), 1: (3, 2, 5, 3, 1) },
            # Env 1  {agent_id: (agent_pos, t_to_move, next_pos), ...}
            { 0: (2, 1, -1, 0, 0), 1: (4, 0, 3, 3, 0), 2: (5, 5, 6, 5, 6) },
        ]
        
        """
        positions = np.argwhere(obs[env_idx, :, :, 11] == 1)
        positions = sorted(positions.tolist(), key=lambda x: (x[0], x[1]))
        pos_dict = {agent_id: tuple(pos) + (-1, 0, 0) for agent_id, pos in enumerate(positions)}
        self.agent_id_pos_map[env_idx] = pos_dict


    def _update_agent_positions(self, action, obs):
        move_delta = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # N, E, S, W
        for env_idx in range(self.env.num_envs):
            for agent_id in list(self.agent_id_pos_map[env_idx].keys()):
                x, y, t, nx, ny = self.agent_id_pos_map[env_idx][agent_id]

                if t == 0: # time to move
                    self.agent_id_pos_map[env_idx][agent_id] = (nx, ny, -1, 0, 0)
                else:
                    self.agent_id_pos_map[env_idx][agent_id] = (x, y, t - 1, nx, ny)
                
                if t < 0 and obs[env_idx, x, y, 22] == 1: # moved issued
                    direction = int(action[env_idx, x, y, 1])
                    dx, dy = move_delta[direction]
                    new_x, new_y = x + dx, y + dy
                    self.agent_id_pos_map[env_idx][agent_id] = (x, y, 8 - 2, new_x, new_y)
        
