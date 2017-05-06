# import numpy as np
# from gym import utils
# from gym.envs.mujoco import mujoco_env
#
# class InvertedPendulumEnv(mujoco_env.MujocoEnv, utils.EzPickle):
#     def __init__(self):
#         utils.EzPickle.__init__(self)
#         mujoco_env.MujocoEnv.__init__(self, 'inverted_pendulum.xml', 2)
#
#         #self.init_qpos = np.array([0., -np.pi])
#
#     def _step(self, a):
#         reward = 1.0
#         self.do_simulation(a, self.frame_skip)
#         ob = self._get_obs()
#         notdone = np.isfinite(ob).all() #and (np.abs(ob[1]) <= .2)
#         done = not notdone
#         return ob, reward, done, {}
#
#     def reset_model(self):
#         qpos = self.init_qpos + self.np_random.uniform(size=self.model.nq, low=-0.01, high=0.01)
#         qvel = self.init_qvel + self.np_random.uniform(size=self.model.nv, low=-0.01, high=0.01)
#         self.set_state(qpos, qvel)
#         return self._get_obs()
#
#     def _get_obs(self):
#         return np.concatenate([self.model.data.qpos, self.model.data.qvel]).ravel()
#
#     def viewer_setup(self):
#         v = self.viewer
#         v.cam.trackbodyid = 0
#         v.cam.distance = v.model.stat.extent
#
# if __name__ == '__main__':
#     env = InvertedPendulumEnv()
#     env.reset()
#     env.render()
#     import IPython; IPython.embed()

import numpy as np

from gym.envs.classic_control.cartpole import CartPoleEnv
from rllab.spaces.discrete import Discrete
from rllab.spaces.box import Box

class CartPoleSwingupEnv(CartPoleEnv):

    def __init__(self, x_threshold=2.4, x_threshold_mult=10):
        CartPoleEnv.__init__(self)

        self.x_threshold = x_threshold
        self.x_threshold_mult = x_threshold_mult

        # Angle limit set to 2 * theta_threshold_radians so failing observation is still within bounds
        high = np.array([
            self.x_threshold * 2,
            np.finfo(np.float32).max,
            1.1,
            1.1,
            np.finfo(np.float32).max])

        self.action_space = Discrete(2)
        self.observation_space = Box(-high, high)

    def _get_obs(self):
        x, x_dot, theta, theta_dot = self.state
        return np.array([x, x_dot, np.cos(theta), np.sin(theta), theta_dot])

    def _step(self, action):
        x, x_dot, theta, theta_dot = self.state

        self.steps_beyond_done = None
        super(CartPoleSwingupEnv, self)._step(action)

        obs = self._get_obs()
        costs = np.power(angle_normalize(theta), 2) + \
                0.1 * np.power(theta_dot, 2) + \
                0.001 * np.dot(action, action)
        done = (np.abs(x) > self.x_threshold)
        if done:
            costs *= self.x_threshold_mult
        reward = -costs

        return obs, reward, done, {}


    def _reset(self):
        # x, x_dot, theta, theta_dot = state
        self.state = np.random.uniform([-0.05, -0.05, np.pi-0.05, -0.05],
                                       [0.05, 0.05, np.pi + 0.05, 0.05])
        self.steps_beyond_done = None
        return self._get_obs()

def angle_normalize(x):
    return (((x+np.pi) % (2*np.pi)) - np.pi)

if __name__ == '__main__':
    env = CartPoleSwingupEnv()
    env.reset()
    env.render()
    import IPython; IPython.embed()
