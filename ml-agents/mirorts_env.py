import numpy as np
from gym_microrts.envs.vec_env import MicroRTSGridModeVecEnv
from mlagents_envs.base_env import ActionSpec, ActionTuple, AgentId, BaseEnv, BehaviorMapping, BehaviorName, BehaviorSpec, DecisionSteps, DimensionProperty, ObservationSpec, ObservationType, TerminalSteps
from typing import Dict, List, Optional, Tuple, Mapping as MappingType

class MicroRTSEnv(BaseEnv):
    def __init__(self, num_selfplay_envs, num_bot_envs, partial_obs = False, max_steps = 2000, render_theme = 2, frame_skip = 0, ai2s = [], map_paths = ["maps/12x12/basesWorkers12x12.xml"], reward_weight = np.array([0.0, 1.0, 0.0, 0.0, 0.0, 5.0]), cycle_maps = [], autobuild = False, jvm_args = []):
        super(MicroRTSEnv, self).__init__()

        self.env = MicroRTSGridModeVecEnv(num_selfplay_envs, num_bot_envs, partial_obs, max_steps, render_theme, frame_skip, ai2s, map_paths, reward_weight, cycle_maps, autobuild, jvm_args)
        
        obs_shape = self.env.observation_space.shape
        self.observation_spec = ObservationSpec(shape = obs_shape, dimension_property = [DimensionProperty.TRANSLATIONAL_EQUIVARIANCE] * len(obs_shape), observation_type = ObservationType.DEFAULT, name = "all")
        self.action_spec = ActionSpec(continuous_size = 0, discrete_branches = self.env.action_space_dims)
        self.behavior_spec = BehaviorSpec(observation_specs = [self.observation_spec], action_spec = self.action_spec)

        self._env_state: Dict[str, Tuple[DecisionSteps, TerminalSteps]] = {}
        self._env_specs: Dict[str, BehaviorSpec] = {}
        self._env_actions: Dict[str, ActionTuple] = {}

    def step(self) -> None:
        # TODO: parse self._env_actions to action
        obs, reward, done, info = self.env.step(action)
        # TODO: save obs, reward, done to self._env_state

    def reset(self) -> None:
        obs = self.env.reset()
        # TODO: save obs to self._env_state

    def close(self):
        self.env.close()

    @property
    def behavior_specs(self) -> MappingType[str, BehaviorSpec]:
        return BehaviorMapping({"default behavior": self.behavior_spec})
    
    def set_actions(self, behavior_name: str, action: ActionTuple) -> None:
        self._env_actions[behavior_name] = action

    def set_action_for_agent(self, behavior_name: str, agent_id: int, action: ActionTuple) -> None:
        if behavior_name not in self._env_actions:
            num_agents = len(self._env_state[behavior_name][0])
            self._env_actions[behavior_name] = self.action_spec.empty_action(num_agents)
        index = np.where(self._env_state[behavior_name][0].agent_id == agent_id)[0][0]
        if self.action_spec.continuous_size > 0:
            self._env_actions[behavior_name].continuous[index] = action.continuous[0, :]
        if self.action_spec.discrete_size > 0:
            self._env_actions[behavior_name].discrete[index] = action.discrete[0, :]

    def get_steps(self, behavior_name: BehaviorName) -> Tuple[DecisionSteps | TerminalSteps]:
        return self._env_state[behavior_name]