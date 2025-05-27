import numpy as np
from gym_microrts.envs.vec_env import MicroRTSGridModeVecEnv

class MultiDiscrete:
    def __init__(self, nvec):
        self.low = np.array([0 for x in nvec])
        self.high = np.array([x for x in nvec])
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
        
        self.observation_space = [self.env.observation_space for _ in range(self.num_agents)]
        self.share_observation_space = [self.env.observation_space for _ in range(self.num_agents)]
        self.action_space = [MultiDiscrete(self.env.action_space.nvec) for _ in range(self.num_agents)]

    def reset(self):
        obs = self.env.reset()
        self.last_obs = obs
        obs = self._obs_wrapper(obs)
        return obs

    def step(self, action):
        # self.env.get_action_mask() (should be done in action selection?)
        self.last_action = action
        action = self._action_wrapper(action)
        obs, reward, done, info = self.env.step(action)
        obs = self._obs_wrapper(obs)
        reward = self._reward_wrapper(reward)
        done = self._done_wrapper(done)
        info = self._info_wrapper(info)
        self.env.render()
        self.last_obs = obs
        return obs, reward, done, info

    def close(self):
        self.env.close()

    # TODO
    def _obs_wrapper(self, obs):
        obs = np.repeat(obs[:, np.newaxis, ...], self.num_agents, axis = 1)
        return obs
    
    # TODO
    def _reward_wrapper(self, reward, new_obs, done):
        last_action = self.last_action
        last_obs = self.last_obs
        agent_positions = self.agent_pos
        w = self.env.width
        h = self.env.height
        
        num_envs, _, _ = last_action.shape
        num_agents = len(agent_positions)
        rewards = np.zeros((num_envs, num_agents), dtype=np.float32)

        for agent_id, pos in enumerate(agent_positions):
            x, y = pos // w, pos % w
            for env in range(num_envs):
                action = last_action[env, agent_id]
                if action[0] != 5:  # not an attack
                    continue

                rel_idx = action[6].item()
                if not (0 <= rel_idx < 49):
                    continue  # invalid relative index

                cx, cy = 3, 3  # center of 7x7
                dx = (rel_idx // 7) - cx
                dy = (rel_idx % 7) - cy
                tx, ty = x + dx, y + dy

                if not (0 <= tx < h and 0 <= ty < w):
                    continue  # target outside map

                attacker_owner = np.argmax(last_obs[env, x, y, 10:13]).item()
                target_owner = np.argmax(last_obs[env, tx, ty, 10:13]).item()

                if target_owner == 0:
                    continue  # no unit there

                if target_owner != attacker_owner:
                    rewards[env, agent_id] += 1  # hit enemy
                else:
                    rewards[env, agent_id] -= 1  # hit ally

        # Win/loss rewards
        for env in range(num_envs):
            if not done[env]:
                continue

            owner_planes = new_obs[env, :, :, 10:13].reshape(-1, 3).sum(dim=0)
            p1_alive = owner_planes[1].item() > 0
            p2_alive = owner_planes[2].item() > 0

            for agent_id in range(num_agents):
                ax, ay = agent_positions[agent_id] // w, agent_positions[agent_id] % w
                agent_owner = np.argmax(last_obs[env, ax, ay, 10:13]).item()

                if agent_owner == 1:
                    if p1_alive and not p2_alive:
                        rewards[env, agent_id] += 10
                    elif not p1_alive and p2_alive:
                        rewards[env, agent_id] -= 10
                elif agent_owner == 2:
                    if p2_alive and not p1_alive:
                        rewards[env, agent_id] += 10
                    elif not p2_alive and p1_alive:
                        rewards[env, agent_id] -= 10

        return rewards
    
    # TODO
    def _done_wrapper(self, done):
        done = np.repeat(done[:, np.newaxis, ...], self.num_agents, axis = 1)
        return done

    # TODO
    def _info_wrapper(self, info):
        return info

    # TODO
    def _action_wrapper(self, action):
        num_envs, num_agents, action_dim = action.shape
        grid_size = self.env.width * self.env.height
        grid_action = np.zeros((num_envs, grid_size * action_dim), dtype=action.dtype)
        for agent_id in range(num_agents):
            pos_index = self.agent_pos[agent_id] # TBD according to agent_pos's shape
            start = pos_index * 7
            end = start + 7
            grid_action[:, start:end] = action[:, agent_id, :]
        return grid_action
    
    def render(self):
        print('render')