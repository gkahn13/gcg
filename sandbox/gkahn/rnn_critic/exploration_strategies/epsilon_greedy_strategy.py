from rllab.core.serializable import Serializable
from rllab.exploration_strategies.base import ExplorationStrategy
from sandbox.rocky.tf.spaces.discrete import Discrete

import numpy as np

class EpsilonGreedyStrategy(ExplorationStrategy, Serializable):
    """
    Takes random action with probability epsilon
    """
    def __init__(self, env_spec, epsilon, decay_func=lambda x: 1):
        assert isinstance(env_spec.action_space, Discrete)
        Serializable.quick_init(self, locals())
        self._env_spec = env_spec
        self._epsilon = epsilon
        if type(decay_func) is str:
            decay_func = eval(decay_func)
        self._decay_func = decay_func

    def get_action(self, t, observation, policy, **kwargs):
        if np.random.random() < self._epsilon * self._decay_func(t):
            action = self._env_spec.action_space.sample()
        else:
            action, _ = policy.get_action(observation)

        return action
