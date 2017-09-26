from rllab.misc.ext import set_seed
### environments
import gym
from sandbox.rocky.tf.envs.base import TfEnv
from rllab.envs.normalized_env import normalize

def create_env(env_str, is_normalize=True, seed=None):
    from rllab.envs.gym_env import GymEnv, FixedIntervalVideoSchedule

    from sandbox.gkahn.rnn_critic.envs.rccar.square_env import SquareEnv
    from sandbox.gkahn.rnn_critic.envs.rccar.square_cluttered_env import SquareClutteredEnv
    from sandbox.gkahn.rnn_critic.envs.rccar.cylinder_env import CylinderEnv

    inner_env = eval(env_str)
    if is_normalize:
        inner_env = normalize(inner_env)
    env = TfEnv(inner_env)

    # set seed
    if seed is not None:
        set_seed(seed)
        if isinstance(inner_env, GymEnv):
            inner_env.env.seed(seed)

    return env
