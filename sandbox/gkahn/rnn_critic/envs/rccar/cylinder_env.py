import os
import numpy as np
import cv2

from sandbox.gkahn.rnn_critic.envs.rccar.car_env import CarEnv

from rllab.spaces.box import Box

class CylinderEnv(CarEnv):

    def __init__(self, params={}):
        params.setdefault('obs_shape', (64, 36)) # not for size b/c don't want aliasing
        params.setdefault('collision_reward_only', False)
        params.setdefault('steer_limits', [-15., 15.])
        params.setdefault('speed_limits', [2., 2.])

        params.setdefault('use_depth', False)
        params.setdefault('do_back_up', False)
        self._model_path = params.get('model_path',
                                      os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models/cylinder.egg'))

        self._obs_shape = params['obs_shape']

        CarEnv.__init__(
            self,
            params=params)

        self.action_space = Box(low=np.array([params['steer_limits'][0], params['speed_limits'][0]]),
                                high=np.array([params['steer_limits'][1], params['speed_limits'][1]]))
        self.observation_space = Box(low=0, high=255, shape=tuple(self._get_observation().shape))

        self._collision_reward_only = params['collision_reward_only']

    ### special for rllab

    def _get_observation(self):
        obs = super(CylinderEnv, self)._get_observation()
        if self._use_depth:
            im = CylinderEnv.process_depth(obs, self._obs_shape)
        else:
            im = CylinderEnv.process_image(obs, self._obs_shape)
            im = np.expand_dims(im, 2)

        return im

    @staticmethod
    def process_depth(image, obs_shape):
        im = np.reshape(image, (image.shape[0], image.shape[1]))
        if im.shape != obs_shape:
            im = cv2.resize(im, obs_shape, interpolation=cv2.INTER_AREA)
        return im.astype(np.uint8)

    @staticmethod
    def process_image(image, obs_shape):
        def rgb2gray(rgb):
            return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])

        image = rgb2gray(image)
        im = cv2.resize(image, obs_shape, interpolation=cv2.INTER_AREA) #TODO how does this deal with aspect ratio
        return im.astype(np.uint8)

    ### default

    def _default_pos(self):
        return (0.0, -6., 0.25)

    def _get_done(self):
        return self._collision or np.array(self._vehicle_pointer.getPos())[1] >= 6.0

    def _get_info(self):
        info = {}
        info['pos'] = np.array(self._vehicle_pointer.getPos())
        info['hpr'] = np.array(self._vehicle_pointer.getHpr())
        info['vel'] = self._get_speed()
        info['coll'] = self._collision and np.array(self._vehicle_pointer.getPos())[1] < 6.0
        return info

    def _default_restart_pos(self):
        ran = np.linspace(-3.5, 3.5, 20)
        np.random.shuffle(ran)
        restart_pos = []
        for val in ran:
            restart_pos.append([val, -6.0, 0.25, 0.0, 0.0, 3.14])
        return restart_pos

    def _get_reward(self):
        if self._collision:
            reward = self._collision_reward
        else:
            if self._collision_reward_only:
                reward = 0
            else:
                reward = self._get_speed()
        return reward

    @property
    def horizon(self):
        return 24

if __name__ == '__main__':
    params = {'visualize': True, 'run_as_task': True}
    env = CylinderEnv(params)
