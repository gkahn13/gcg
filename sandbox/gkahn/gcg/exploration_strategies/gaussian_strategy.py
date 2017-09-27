import numpy as np

from rllab.core.serializable import Serializable
from rllab.exploration_strategies.base import ExplorationStrategy
from sandbox.rocky.tf.spaces.box import Box
from sandbox.gkahn.gcg.utils import schedules

class GaussianStrategy(ExplorationStrategy, Serializable):
    """
    Add gaussian noise
    """
    def __init__(self, env_spec, endpoints, outside_value):
        assert isinstance(env_spec.action_space, Box)
        Serializable.quick_init(self, locals())
        self._env_spec = env_spec
        self.schedule = schedules.PiecewiseSchedule(endpoints=endpoints, outside_value=outside_value)

    def reset(self):
        pass

    def add_exploration(self, t, action):
        return np.clip(action + np.random.normal(size=len(action)) * self.schedule.value(t),
                       self._env_spec.action_space.low, self._env_spec.action_space.high)
