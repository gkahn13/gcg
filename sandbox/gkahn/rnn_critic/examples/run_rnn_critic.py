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
### exploration strategies
from sandbox.gkahn.rnn_critic.exploration_strategies.gaussian_strategy import GaussianStrategy
from sandbox.gkahn.rnn_critic.exploration_strategies.epsilon_greedy_strategy import EpsilonGreedyStrategy
### RNN critic
from sandbox.gkahn.rnn_critic.algos.rnn_critic import RNNCritic
from sandbox.gkahn.rnn_critic.algos.rnn_critic_offpolicy import RNNCriticOffpolicy
from sandbox.gkahn.rnn_critic.policies.dqn_policy import DQNPolicy
from sandbox.gkahn.rnn_critic.policies.discrete_dqn_policy import DiscreteDQNPolicy
from sandbox.gkahn.rnn_critic.policies.nstep_dqn_policy import NstepDQNPolicy
from sandbox.gkahn.rnn_critic.policies.nstep_discrete_dqn_policy import NstepDiscreteDQNPolicy
from sandbox.gkahn.rnn_critic.policies.multiaction_combinedcost_mlp_policy import MultiactionCombinedcostMLPPolicy
from sandbox.gkahn.rnn_critic.policies.multiaction_combinedcost_rnn_policy import MultiactionCombinedcostRNNPolicy
from sandbox.gkahn.rnn_critic.policies.multiaction_combinedcost_muxrnn_policy import MultiactionCombinedcostMuxRNNPolicy
from sandbox.gkahn.rnn_critic.policies.multiaction_separatedcost_mlp_policy import MultiactionSeparatedcostMLPPolicy
from sandbox.gkahn.rnn_critic.policies.multiaction_separatedcost_rnn_policy import MultiactionSeparatedcostRNNPolicy
### RNN analyze
from sandbox.gkahn.rnn_critic.examples.analyze_experiment import AnalyzeRNNCritic

def run_rnn_critic(params, params_txt):
    # copy yaml for posterity
    yaml_path = os.path.join(logger.get_snapshot_dir(), '{0}.yaml'.format(params['exp_name']))
    with open(yaml_path, 'w') as f:
        f.write(params_txt)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(params['policy']['gpu_device'])  # TODO: hack so don't double GPU
    config.USE_TF = True

    from rllab.envs.gym_env import GymEnv
    from sandbox.gkahn.rnn_critic.envs.premade_gym_env import PremadeGymEnv
    try:
        import gym_ple
    except:
        pass
    from sandbox.gkahn.rnn_critic.envs.point_env import PointEnv
    from sandbox.gkahn.rnn_critic.envs.sparse_point_env import SparsePointEnv
    from sandbox.gkahn.rnn_critic.envs.chain_env import ChainEnv
    inner_env = eval(params['alg'].pop('env'))
    env = TfEnv(normalize(inner_env))

    # set seed
    if params['seed'] is not None:
        set_seed(params['seed'])
        if isinstance(inner_env, GymEnv):
            inner_env.env.seed(params['seed'])

    # import matplotlib.pyplot as plt
    # f = plt.figure()
    # done = True
    # while True:
    #     if done:
    #         o = env.reset()
    #         r = 0
    #         done = False
    #     else:
    #         o, r, done, _ = env.step(0)
    #
    #     plt.imshow(o[:,:,0], cmap='Greys_r')
    #     plt.show(block=False)
    #     plt.pause(0.05)
    #     input('done: {0}, r: {1}'.format(done, r))
    #     plt.clf()
    #     plt.cla()
    #
    # import IPython; IPython.embed()

    #####################
    ### Create policy ###
    #####################

    policy_class = params['policy']['class']
    PolicyClass = eval(policy_class)
    policy_params = params['policy'][policy_class]
    get_action = params['get_action']
    get_target_action = params['get_target_action'] if 'get_target_action' in params else params['get_action']
    get_action_type = get_action['type']
    get_target_action_type = get_target_action['type']

    policy = PolicyClass(
        env_spec=env.spec,
        get_action_params=get_action[get_action_type],
        get_target_action_params=get_target_action[get_target_action_type],
        **policy_params,
        **params['policy']
    )

    ###################################
    ### Create exploration strategy ###
    ###################################

    if 'exploration_strategy' in params['alg']:
        es_params = params['alg'].pop('exploration_strategy')
        es_class = es_params['class']
        ESClass = eval(es_class)
        exploration_strategy = ESClass(env_spec=env.spec, **es_params[es_class])
    else:
        exploration_strategy = None

    ########################
    ### Create algorithm ###
    ########################

    if 'is_onpolicy' not in params['alg'].keys() or params['alg']['is_onpolicy']:
        algo = RNNCritic(
            env=env,
            policy=policy,
            exploration_strategy=exploration_strategy,
            max_path_length=params['alg'].pop('max_path_length', env.horizon),
            **params['alg']
        )
    else:
        algo = RNNCriticOffpolicy(
            env=env,
            policy=policy,
            **params['alg']
        )
    algo.train()

    ###############
    ### Analyze ###
    ###############

    import traceback
    logger.log('Analyzing experiment {0}'.format(logger.get_snapshot_dir()))
    try:
        analyze = AnalyzeRNNCritic(logger.get_snapshot_dir())
        analyze.run()
    except:
        logger.log(traceback.format_exc())
