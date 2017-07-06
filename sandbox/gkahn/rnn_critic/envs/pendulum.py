import numpy as np

from rllab.spaces.discrete import Discrete

from gym.envs.classic_control.pendulum import PendulumEnv, angle_normalize
from gym import spaces

class PendulumContinuousDense(PendulumEnv):
    class spec:
        id = 'PendulumContinuousDense'
        timestep_limit = 200
        

class PendulumContinuousSparse(PendulumEnv):
    class spec:
        id = 'PendulumContinuousSparse'
        timestep_limit = 200

    def __init__(self, theta_sparse=40.):
        PendulumEnv.__init__(self)
        self._theta_sparse = theta_sparse

    def _step(self, u):
        th, thdot = self.state # th := theta                                                                       
        g = 10.
        m = 1.
        l = 1.
        dt = self.dt

        u = np.clip(u, -self.max_torque, self.max_torque)[0]
        self.last_u = u # for rendering                                                                            
        if np.abs(angle_normalize(th)) > (self._theta_sparse * np.pi / 180.):
            costs = np.pi**2 #+ .1*thdot**2 + 0.001*(u**2)
        else:
            costs = angle_normalize(th)**2 + .1*thdot**2 + .001*(u**2)

        newthdot = thdot + (-3*g/(2*l) * np.sin(th + np.pi) + 3./(m*l**2)*u) * dt
        newth = th + newthdot*dt
        newthdot = np.clip(newthdot, -self.max_speed, self.max_speed) #pylint: disable=E1111                       
        self.state = np.array([newth, newthdot])
        return self._get_obs(), -costs, False, {}


class PendulumDiscreteDense(PendulumEnv):
    class spec:
        id = 'PendulumDiscreteDense'
        timestep_limit = 200

    def __init__(self):
        PendulumEnv.__init__(self)
        self.action_space = spaces.Discrete(9)

    def _step(self, u):
        u_cont = np.array([np.linspace(-self.max_torque, self.max_torque, self.action_space.n)[u]])
        return super(PendulumDiscreteDense, self)._step(u_cont)

class PendulumDiscreteSparse(PendulumContinuousSparse, PendulumDiscreteDense):
    class spec:
        id = 'PendulumDiscreteSparse'
        timestep_limit = 200

    def __init__(self):
        PendulumContinuousSparse.__init__(self, theta_sparse=90.)
        PendulumDiscreteDense.__init__(self)

    def _step(self, u):
        u_cont = np.array([np.linspace(-self.max_torque, self.max_torque,self.action_space.n)[u]])
        return PendulumContinuousSparse._step(self, u_cont)
    
