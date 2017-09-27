import os

from sandbox.gkahn.gcg.envs.rccar.cylinder_env import CylinderEnv

class SquareEnv(CylinderEnv):

    def __init__(self, params={}):
        params.setdefault('do_back_up', True)
        params.setdefault('back_up', {
            'steer': [-5., 5.],
            'vel': -1.,
            'duration': 3.
        })
        params.setdefault('model_path', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models/square.egg'))

        CylinderEnv.__init__(self, params=params)

    def _setup_light(self):
        pass
        
    def _default_pos(self):
        return (20.0, -19., 0.25)

    def _default_restart_pos(self):
        return [
                [ 20., -20., 0.3, 0.0, 0.0, 3.14],
                [-20., -20., 0.3, 0.0, 0.0, 3.14],
                [ 20.,  15., 0.3, 0.0, 0.0, 3.14],
                [-20.,  15., 0.3, 0.0, 0.0, 3.14]
            ]

    def _get_done(self):
        return self._collision

    @property
    def horizon(self):
        # at 2m/s, roughly 80 steps per side
        return int(1e3)

if __name__ == '__main__':
    params = {'visualize': True, 'run_as_task': True, 'do_back_up': True, 'hfov': 120}
    env = SquareEnv(params)
