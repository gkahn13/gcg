import os

import rllab.misc.logger as logger
from rllab import config

### environments
from sandbox.gkahn.rnn_critic.envs.env_utils import create_env

### RNN critic
from sandbox.gkahn.rnn_critic.algos.rnn_critic import RNNCritic
from sandbox.gkahn.rnn_critic.policies.mac_policy import MACPolicy
from sandbox.gkahn.rnn_critic.policies.rccar_mac_policy import RCcarMACPolicy

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

    normalize_env = params['alg'].pop('normalize_env')

    env_str = params['alg'].pop('env')
    env = create_env(env_str, is_normalize=normalize_env, seed=params['seed'])

    env_eval_str = params['alg'].pop('env_eval', env_str)
    env_eval = create_env(env_eval_str, is_normalize=normalize_env, seed=params['seed'])

    env.reset()
    env_eval.reset()

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
