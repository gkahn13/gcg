import os
import numpy as np

from sandbox.gkahn.rnn_critic.envs.rccar.car_env import CarEnv

from rllab.spaces.box import Box

class SquareEnv(CarEnv):

    def __init__(self, params={}):
        params.setdefault('size', [64, 36])
        params.setdefault('use_depth', True)
        params.setdefault('do_back_up', True)
        params.setdefault('back_up', {
            'steer': [-5., 5.],
            'vel': -1.,
            'duration': 3.
        })
        self._model_path = params.get('model_path',
                                      os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models/square.egg'))
        CarEnv.__init__(
            self,
            params=params)

        self.action_space = Box(low=np.array([-45., -1.]), high=np.array([45., 4.]))
        self.observation_space = Box(low=0, high=255, shape=tuple(self._get_observation().shape))

    def _setup_light(self):
        pass
        
    def _default_pos(self):
        return (20.0, -19., 0.25)

    def _get_reward(self):
        reward = 0 if self._collision else self._get_speed()
        return reward

    def _default_restart_pos(self):
        return [
                [ 20., -20., 0.3, 0.0, 0.0, 3.14],
                [-20., -20., 0.3, 0.0, 0.0, 3.14],
                [ 20.,  15., 0.3, 0.0, 0.0, 3.14],
                [-20.,  15., 0.3, 0.0, 0.0, 3.14]
            ]

    @property
    def horizon(self):
        return int(1e4)

if __name__ == '__main__':
    # params = {'visualize': True, 'run_as_task': True, 'model_path': 'models/square.egg'}
    params = {'visualize': True, 'run_as_task': True}
    env = SquareEnv(params)
    import IPython; IPython.embed()
