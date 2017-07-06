#!/usr/bin/env python
import os
from sandbox.gkahn.rnn_critic.envs.sim_rccar.car_env import CarEnv

class SquareEnv(CarEnv):
    
    def __init__(self, params={}):
        self._model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models/square_hallway.egg')
        params.update({
            'do_back_up': params.get('do_back_up', True),
            'back_up': {
                'cmd_steer': [-5., 5.],
                'cmd_vel': -8.,
                'duration': 3.0,
            }
        })
        CarEnv.__init__(
            self,
            params=params)

    def _default_pos(self):
        return (42.5, -42.5, 0.2)

    @property
    def horizon(self):
        return int(1e4)
