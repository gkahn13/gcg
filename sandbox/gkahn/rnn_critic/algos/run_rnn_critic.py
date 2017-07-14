import os, argparse, yaml, shutil

import numpy as np
import random
import tensorflow as tf

from rllab.misc.instrument import run_experiment_lite
import rllab.misc.logger as logger
from rllab.misc.ext import set_seed
from rllab import config
### environments
from sandbox.gkahn.rnn_critic.envs.env_utils import create_env
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
from sandbox.gkahn.rnn_critic.policies.nstep_mac_policy import NstepMACPolicy
from sandbox.gkahn.rnn_critic.policies.nstep_mac_mux_policy import NstepMACMuxPolicy
from sandbox.gkahn.rnn_critic.policies.discrete_mac_policy import DiscreteMACPolicy
from sandbox.gkahn.rnn_critic.policies.random_discrete_mac_policy import RandomDiscreteMACPolicy

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

    env_str = params['alg'].pop('env')
    env = create_env(env_str, seed=params['seed'])

    env_eval_str = params['alg'].pop('env_eval', env_str)
    env_eval = create_env(env_eval_str, seed=params['seed'])

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
        exploration_strategies=params['alg'].pop('exploration_strategies'),
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
        env_str=env_str,
        **params['alg']
    )
    algo.train()
