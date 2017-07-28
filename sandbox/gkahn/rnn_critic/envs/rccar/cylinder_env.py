import os
import numpy as np

from sandbox.gkahn.rnn_critic.envs.rccar.car_env import CarEnv

from rllab.spaces.box import Box

class CylinderEnv(CarEnv):

    def __init__(self, params={}):
        params.setdefault('size', [64, 36])
        params.setdefault('use_depth', False)
        params.setdefault('do_back_up', False)
        params.setdefault('collision_reward', 0.)
        self._model_path = params.get('model_path',
                                      os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models/cylinder.egg'))
        CarEnv.__init__(
            self,
            params=params)

        self.action_space = Box(low=np.array([-45., 0.]), high=np.array([45., 2.]))
        self.observation_space = Box(low=0, high=255, shape=tuple(self._get_observation().shape))
        self._collision_reward = params['collision_reward']


    def _default_pos(self):
        return (0.0, -6., 0.25)

    def _default_restart_pos(self):
        ran = np.linspace(-3.5, 3.5, 20)
        np.random.shuffle(ran)
        restart_pos = []
        for val in ran:
            restart_pos.append([val, -6.0, 0.25, 0.0, 0.0, 3.14])
        return restart_pos

    def _get_reward(self):
        reward = self._collision_reward if self._collision else self._get_speed()
        return reward

    def _get_observation(self):
        obs = super(CylinderEnv, self)._get_observation()
        if self._use_depth:
            return obs
        else:
            return np.expand_dims(obs[:, :, 0] * 0.299 + obs[:, :, 1] * 0.587 + obs[:, :, 2] * 0.114, 2).astype(np.uint8)

    @property
    def horizon(self):
        return 24


if __name__ == '__main__':
    params = {'visualize': True, 'run_as_task': False}
    env = CylinderEnv(params)
    import IPython; IPython.embed()
