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

class CartPoleSwingupEnv(CartPoleEnv):

    def __init__(self):
        CartPoleEnv.__init__(self)

    def _step(self, action):
        x, x_dot, theta, theta_dot = self._state
        obs, reward, done, info = super(CartPoleSwingupEnv, self)._step(action)


    def _reset(self):
        # x, x_dot, theta, theta_dot = state
        self.state = np.random.uniform([-0.05, -0.05, np.pi-0.05, -0.05],
                                       [0.05, 0.05, np.pi + 0.05, 0.05])
        self.steps_beyond_done = None
        return np.array(self.state)

if __name__ == '__main__':
    env = CartPoleSwingupEnv()
    env.reset()
    env.render()
    import IPython; IPython.embed()
