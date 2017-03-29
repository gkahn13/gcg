import os, argparse, yaml, shutil

import numpy as np
import random
import tensorflow as tf

from rllab.misc.instrument import run_experiment_lite
import rllab.misc.logger as logger
from rllab.misc.ext import set_seed
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

def run_task(params):
    # copy yaml for posterity
    shutil.copy(params['yaml_path'], os.path.join(logger.get_snapshot_dir(), os.path.basename(params['yaml_path'])))

    from rllab.envs.gym_env import GymEnv
    from sandbox.gkahn.rnn_critic.envs.premade_gym_env import PremadeGymEnv
    import gym_ple
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

    #####################
    ### Create policy ###
    #####################

    policy_class = params['policy']['class']
    PolicyClass = eval(policy_class)
    policy_params = params['policy'][policy_class]
    get_action_type = params['get_action']['type']

    policy = PolicyClass(
        env_spec=env.spec,
        get_action_params=params['get_action'][get_action_type],
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
            max_path_length=env.horizon,
            **params['alg']
        )
    else:
        algo = RNNCriticOffpolicy(
            env=env,
            policy=policy,
            **params['alg']
        )
    algo.train()


def main(yaml_file):
    assert (os.path.exists(yaml_file))
    with open(yaml_file, 'r') as f:
        params = yaml.load(f)
    params['yaml_path'] = yaml_file

    os.environ["CUDA_VISIBLE_DEVICES"] = str(params['policy']['gpu_device']) # TODO: hack so don't double GPU

    run_experiment_lite(
        lambda x: run_task(params), # HACK
        snapshot_mode="all",
        exp_name=params['exp_name'],
        exp_prefix=params['exp_prefix'],
        # use_gpu=True,
        # mode='local_docker',
        # docker_image='rllab-gkahn'
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('yaml', type=str)
    args = parser.parse_args()

    main(args.yaml)
