import numpy as np

from rllab.core.serializable import Serializable
from sandbox.rocky.tf.spaces.box import Box

class OUStrategy(Serializable):
    """
    Add Ornstein-Uhlenbeck noise
    """
    def __init__(self, env_spec, mu, theta, sigma):
        assert isinstance(env_spec.action_space, Box)
        Serializable.quick_init(self, locals())
        self._env_spec = env_spec
        self._mu = mu
        self._theta = theta
        self._sigma = sigma

        self._state = None
        self.reset()

    def _evolve_state(self):
        self._state = self._theta * (self._mu - self._state) + self._sigma * np.random.randn(len(self._state))
        return self._state

    def reset(self):
        self._state = np.ones(self._env_spec.action_space.flat_dim) * self._mu

    def add_exploration(self, t, action):
        ou_state = self._evolve_state()
        return np.clip(action + ou_state, self._env_spec.action_space.low, self._env_spec.action_space.high)
