import numpy as np

from sandbox.gkahn.rnn_critic.envs.sparse_point_env import SparsePointEnv

class PointEnv(SparsePointEnv):
    def __init__(self):
        SparsePointEnv.__init__(self, np.inf)
