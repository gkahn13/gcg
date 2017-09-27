from rllab.core.serializable import Serializable
from sandbox.gkahn.gcg.utils import schedules

import numpy as np

class EpsilonGreedyStrategy(Serializable):
    """
    Takes random action with probability epsilon
    """
    def __init__(self, env_spec, endpoints, outside_value):
        Serializable.quick_init(self, locals())
        self._env_spec = env_spec
        self.schedule = schedules.PiecewiseSchedule(endpoints=endpoints, outside_value=outside_value)

    def reset(self):
        pass

    def add_exploration(self, t, action):
        if np.random.random() < self.schedule.value(t):
            return self._env_spec.action_space.sample()
        else:
            return action
