from rllab.envs.base import Env
from rllab.spaces.discrete import Discrete
from rllab.spaces.box import Box
from rllab.envs.base import Step

import numpy as np

class PhdEnv(Env):
    CONTINUE = 0
    ESCAPE = 1

    @staticmethod
    def r_thesis_to_finish(r_continue, r_escape, gamma, length):
        return r_escape - r_continue * (1 - np.power(gamma, length)) / ((1 - gamma) * np.power(gamma, length))

    def __init__(self, length=10, r_continue=-1, r_escape=1, r_thesis=12):
        """
        [0, 1, ..., length-1, escape terminal, thesis terminal]
        """
        self._length = length
        self._num_states = length + 2

        self._r_continue = r_continue
        self._r_escape = r_escape
        self._r_thesis = r_thesis

    def _state_to_observation(self, state):
        # observation = np.zeros(self._num_states, dtype=np.float64)
        # observation[state] = 1.
        # return observation
        observation = np.array([2 * ((float(state) / float(self._num_states)) - 0.5)])
        return observation

    @property
    def observation_space(self):
        # return Box(0, 1, (self._num_states,))
        return Box(-1, 1, (1,))

    @property
    def action_space(self):
        return Discrete(2) # continue or escape

    def reset(self):
        self._state = 0
        observation = self._state_to_observation(self._state)
        return observation

    def step(self, action):
        assert(action == PhdEnv.ESCAPE or action == PhdEnv.CONTINUE)

        observation = self._state_to_observation(self._state)
        done = (self._state >= self._length)
        if done:
            reward = 0
        else:
            is_escape = (action == PhdEnv.ESCAPE)
            is_thesis = (action == PhdEnv.CONTINUE and self._state == self._length - 1)

            if is_escape:
                self._state = self._num_states - 2
                reward = self._r_escape
            elif is_thesis:
                self._state = self._num_states - 1
                reward = self._r_thesis
            else:
                self._state += 1
                reward = self._r_continue

        return Step(observation=observation, reward=reward, done=done)

    def render(self):
        print('Current state: {0:d}'.format(self._state))

    @property
    def horizon(self):
        return 10 * self._length

    def get_param_values(self):
        return dict()

    def set_param_values(self, params):
        pass
