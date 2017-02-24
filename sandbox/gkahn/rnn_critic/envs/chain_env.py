from rllab.envs.base import Env
from rllab.spaces.discrete import Discrete
from rllab.envs.base import Step

import numpy as np

class ChainEnv(Env):
    def __init__(self, length):
        """
        :param length: length of chain
        """
        self._length = length

    def _state_to_observation(self, state):
        observation = np.zeros(self._length, dtype=int)
        observation[state] = 1.
        return observation

    @property
    def observation_space(self):
        return Discrete(self._length)

    @property
    def action_space(self):
        return Discrete(3) # left stay right

    def reset(self):
        self._state = 0
        # observation = self._state_to_observation(self._state)
        observation = self._state
        return observation

    def step(self, action):
        done = (self._state == self._length - 1)
        if done:
            reward = 1.
        else:
            if (self._state == self._length - 2) and action == 2:
                reward = 1.
            elif action == 1:
                reward = 0.
            else:
                reward = -0.1 / float(self._length)

        assert(type(action) is int)
        assert(self._state >= 0 and self._state < self._length)
        if self._state > 0 and self._state < self._length - 1 and action == 0:
            self._state -= 1
        elif self._state < self._length - 1 and action == 2:
            self._state += 1

        # observation = self._state_to_observation(self._state)
        observation = self._state

        return Step(observation=observation, reward=reward, done=done)

    def render(self):
        print('Current state: {0:d}'.format(self._state[0]))

    @property
    def horizon(self):
        return 10 * self._length

    def get_param_values(self):
        return dict()

    def set_param_values(self, params):
        pass
