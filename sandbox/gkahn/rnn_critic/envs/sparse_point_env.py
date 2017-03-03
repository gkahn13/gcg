from rllab.envs.base import Env
from rllab.spaces import Box
from rllab.envs.base import Step
import numpy as np

class SparsePointEnv(Env):
    def __init__(self, radius):
        self._radius = radius

    @property
    def observation_space(self):
        return Box(low=-np.inf, high=np.inf, shape=(2,))

    @property
    def action_space(self):
        return Box(low=-0.1, high=0.1, shape=(2,))

    def reset(self):
        self._state = np.random.uniform(-1, 1, size=(2,))
        observation = np.copy(self._state)
        return observation

    def step(self, action):
        self._state = self._state + action
        x, y = self._state
        dist = (x ** 2 + y ** 2) ** 0.5
        reward = np.clip(-dist, -self._radius, 0.)
        done = abs(x) < 0.01 and abs(y) < 0.01
        next_observation = np.copy(self._state)
        return Step(observation=next_observation, reward=reward, done=done)

    def render(self):
        print('current state:', self._state)

    @property
    def horizon(self):
        return 50

    def get_param_values(self):
        return dict()

    def set_param_values(self, params):
        pass

    ################################
    ### FOR DEBUGGING/EVALUATING ###
    ################################

    def set_state(self, state):
        self._state = np.copy(state)
