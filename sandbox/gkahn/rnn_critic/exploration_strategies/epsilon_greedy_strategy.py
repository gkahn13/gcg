from rllab.core.serializable import Serializable
from rllab.exploration_strategies.base import ExplorationStrategy
from sandbox.rocky.tf.spaces.discrete import Discrete
from sandbox.gkahn.rnn_critic.utils import schedules

import numpy as np

class EpsilonGreedyStrategy(ExplorationStrategy, Serializable):
    """
    Takes random action with probability epsilon
    """
    def __init__(self, env_spec, endpoints, outside_value):
        assert isinstance(env_spec.action_space, Discrete)
        Serializable.quick_init(self, locals())
        self._env_spec = env_spec
        self._schedule = schedules.PiecewiseSchedule(endpoints=endpoints, outside_value=outside_value)

    def get_action(self, t, observation, policy, **kwargs):
        if np.random.random() < self._schedule.value(t):
            action = self._env_spec.action_space.sample()
        else:
            action, _ = policy.get_action(observation)

        return action