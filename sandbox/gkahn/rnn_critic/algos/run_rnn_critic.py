import os, argparse, yaml, shutil

import numpy as np
import random
import tensorflow as tf

from rllab.misc.instrument import run_experiment_lite
import rllab.misc.logger as logger
from rllab.misc.ext import set_seed
from rllab import config
### environments
import gym
from sandbox.rocky.tf.envs.base import TfEnv
from rllab.envs.normalized_env import normalize
from sandbox.gkahn.rnn_critic.envs.atari_wrappers import wrap_deepmind
from sandbox.gkahn.rnn_critic.envs.pygame_wrappers import wrap_pygame
### RNN critic
from sandbox.gkahn.rnn_critic.algos.rnn_critic import RNNCritic
from sandbox.gkahn.rnn_critic.policies.mac_policy import MACPolicy
from sandbox.gkahn.rnn_critic.policies.mac_mux_policy import MACMuxPolicy
from sandbox.gkahn.rnn_critic.policies.dqn_policy import DQNPolicy
from sandbox.gkahn.rnn_critic.policies.cdqn_policy import CDQNPolicy
from sandbox.gkahn.rnn_critic.policies.feedforward_mac_policy import FeedforwardMACPolicy
from sandbox.gkahn.rnn_critic.policies.random_mac_policy import RandomMACPolicy
from sandbox.gkahn.rnn_critic.policies.random_mac_mux_policy import RandomMACMuxPolicy
from sandbox.gkahn.rnn_critic.policies.notarget_mac_policy import NotargetMACPolicy
from sandbox.gkahn.rnn_critic.policies.final_mac_policy import FinalMACPolicy

def run_rnn_critic(params):
    # copy yaml for posterity
    try:
        yaml_path = os.path.join(logger.get_snapshot_dir(), '{0}.yaml'.format(params['exp_name']))
        with open(yaml_path, 'w') as f:
            f.write(params['txt'])
    except:
        pass

    os.environ["CUDA_VISIBLE_DEVICES"] = str(params['policy']['gpu_device'])  # TODO: hack so don't double GPU
    config.USE_TF = True

    def create_env(env_str):
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
        from rllab.envs.mujoco.swimmer_env import SwimmerEnv
        from sandbox.gkahn.rnn_critic.envs.point_env import PointEnv
        from sandbox.gkahn.rnn_critic.envs.sparse_point_env import SparsePointEnv
        from sandbox.gkahn.rnn_critic.envs.chain_env import ChainEnv
        from sandbox.gkahn.rnn_critic.envs.phd_env import PhdEnv
        from sandbox.gkahn.rnn_critic.envs.cartpole_swingup_env import CartPoleSwingupEnv, CartPoleSwingupImageEnv

        inner_env = eval(env_str)
        env = TfEnv(normalize(inner_env))

        # set seed
        if params['seed'] is not None:
            set_seed(params['seed'])
            if isinstance(inner_env, GymEnv):
                inner_env.env.seed(params['seed'])

        return env

    env_str = params['alg'].pop('env')
    env = create_env(env_str)

    env_eval_str = params['alg'].pop('env_eval', env_str)
    env_eval = create_env(env_eval_str)

    env.reset()
    env_eval.reset()

    # import matplotlib.pyplot as plt
    # f = plt.figure()
    # done = True
    # im = None
    # while True:
    #     if done:
    #         o = env.reset()
    #         r = 0
    #         done = False
    #     else:
    #         o, r, done, _ = env.step(0)
    #
    #     if im is None:
    #         im = plt.imshow(o[:,:,0], cmap='Greys_r')
    #         plt.show(block=False)
    #     else:
    #         im.set_array(o[:,:,0])
    #     f.canvas.draw()
    #     plt.pause(0.01)
    #     input('done: {0}, r: {1}'.format(done, r))
    #
    # import IPython; IPython.embed()

    #####################
    ### Create policy ###
    #####################

    policy_class = params['policy']['class']
    PolicyClass = eval(policy_class)
    policy_params = params['policy'][policy_class]

    policy = PolicyClass(
        env_spec=env.spec,
        exploration_strategy=params['alg'].pop('exploration_strategy'),
        **policy_params,
        **params['policy']
    )

    ########################
    ### Create algorithm ###
    ########################

    if 'max_path_length' in params['alg']:
        max_path_length = params['alg'].pop('max_path_length')
    else:
        max_path_length = env.horizon
    algo = RNNCritic(
        env=env,
        env_eval=env_eval,
        policy=policy,
        max_path_length=max_path_length,
        **params['alg']
    )
    algo.train()
