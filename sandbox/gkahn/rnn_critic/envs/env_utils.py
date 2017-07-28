from rllab.misc.ext import set_seed
### environments
import gym
from sandbox.rocky.tf.envs.base import TfEnv
from rllab.envs.normalized_env import normalize
from sandbox.gkahn.rnn_critic.envs.atari_wrappers import wrap_deepmind
from sandbox.gkahn.rnn_critic.envs.pygame_wrappers import wrap_pygame

def create_env(env_str, seed=None):
    from rllab.envs.gym_env import GymEnv, FixedIntervalVideoSchedule
    from sandbox.gkahn.rnn_critic.envs.premade_gym_env import PremadeGymEnv
    try:
        import gym_ple
    except:
        pass
    try:
        from sandbox.gkahn.rnn_critic.envs.car.collision_car_racing_env import CollisionCarRacingSteeringEnv, \
            CollisionCarRacingDiscreteEnv
    except:
        pass
    try:
        from sandbox.gkahn.rnn_critic.envs.rccar.square_env import SquareEnv
        from sandbox.gkahn.rnn_critic.envs.rccar.cylinder_env import CylinderEnv
    except:
        pass
    from rllab.envs.mujoco.swimmer_env import SwimmerEnv
    from sandbox.gkahn.rnn_critic.envs.point_env import PointEnv
    from sandbox.gkahn.rnn_critic.envs.sparse_point_env import SparsePointEnv
    from sandbox.gkahn.rnn_critic.envs.chain_env import ChainEnv
    from sandbox.gkahn.rnn_critic.envs.phd_env import PhdEnv
    from sandbox.gkahn.rnn_critic.envs.cartpole_swingup_env import CartPoleSwingupEnv, CartPoleSwingupImageEnv
    from sandbox.gkahn.rnn_critic.envs.pendulum import PendulumContinuousDense, PendulumContinuousSparse, \
        PendulumDiscreteDense, PendulumDiscreteSparse, PendulumStochastic

    inner_env = eval(env_str)
    env = TfEnv(normalize(inner_env))

    # set seed
    if seed is not None:
        set_seed(seed)
        if isinstance(inner_env, GymEnv):
            inner_env.env.seed(seed)

    return env
