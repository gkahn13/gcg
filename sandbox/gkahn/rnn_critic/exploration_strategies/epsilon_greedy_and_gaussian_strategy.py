import numpy as np

from rllab.core.serializable import Serializable
from rllab.exploration_strategies.base import ExplorationStrategy
from sandbox.rocky.tf.spaces.box import Box
from sandbox.gkahn.rnn_critic.utils import schedules

class EpsilonGreedyAndGaussianStrategy(ExplorationStrategy, Serializable):
    """
    Do epsilon greedy or add gaussian noise
    """
    def __init__(self, env_spec,
                 epsilon_greedy_endpoints, epsilon_greedy_outside_value,
                 gaussian_endpoints, gaussian_outside_value):
        assert isinstance(env_spec.action_space, Box)
        Serializable.quick_init(self, locals())
        self._env_spec = env_spec
        self.epsilon_greedy_schedule = schedules.PiecewiseSchedule(endpoints=epsilon_greedy_endpoints,
                                                                   outside_value=epsilon_greedy_outside_value)
        self.gaussian_schedule = schedules.PiecewiseSchedule(endpoints=gaussian_endpoints,
                                                             outside_value=gaussian_outside_value)

    def reset(self):
        pass

    def add_exploration(self, t, action):
        if np.random.random() < self.epsilon_greedy_schedule.value(t):
            return self._env_spec.action_space.sample()
        else:
            return np.clip(action + np.random.normal(size=len(action)) * self.gaussian_schedule.value(t),
                           self._env_spec.action_space.low, self._env_spec.action_space.high)
