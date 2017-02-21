import os, argparse, yaml, shutil
import tensorflow as tf

from rllab.misc.instrument import run_experiment_lite
import rllab.misc.logger as logger
### environments
from sandbox.rocky.tf.envs.base import TfEnv
from rllab.envs.normalized_env import normalize
### exploration strategies
from rllab.exploration_strategies.gaussian_strategy import GaussianStrategy
### RNN critic
from sandbox.gkahn.rnn_critic.algos.rnn_critic import RNNCritic
from sandbox.gkahn.rnn_critic.policies.mlp_policy import RNNCriticMLPPolicy
from sandbox.gkahn.rnn_critic.policies.rnn_policy import RNNCriticRNNPolicy

### parameters loaded from yaml
params = dict()

def run_task(*_):
    # copy yaml for posterity
    shutil.copy(params['yaml_path'], os.path.join(logger.get_snapshot_dir(), os.path.basename(params['yaml_path'])))

    from rllab.envs.gym_env import GymEnv
    from sandbox.gkahn.rnn_critic.envs.point_env import PointEnv
    env = TfEnv(normalize(eval(params['alg'].pop('env'))))

    def create_policy(is_train):
        policy_type = params['policy']['type']
        get_action_type = params['get_action']['type']

        if policy_type == 'mlp':
            PolicyClass = RNNCriticMLPPolicy
        elif policy_type == 'rnn':
            PolicyClass = RNNCriticRNNPolicy
        else:
            raise Exception('Policy {0} not valid'.format(policy_type))

        return PolicyClass(
            is_train=is_train,
            env_spec=env.spec,
            get_action_params=params['get_action'][get_action_type],
            **params['policy'][policy_type],
            **params['policy']
        )

    policy = create_policy(is_train=True)

    es_params = params['alg'].pop('exploration_strategy')
    es_type = es_params['type']
    if es_type == 'gaussian':
        ESClass = GaussianStrategy
    else:
        raise Exception('Exploration strategy {0} not valid'.format(es_type))

    exploration_strategy = ESClass(env_spec=env.spec, **es_params[es_type])

    algo = RNNCritic(
        env=env,
        policy=policy,
        exploration_strategy=exploration_strategy,
        max_path_length=env.horizon,
        **params['alg']
    )
    algo.train()


def main():
    seed = params['seed']
    tf.set_random_seed(seed)

    run_experiment_lite(
        run_task,
        snapshot_mode="all",
        seed=params['seed'],
        exp_name=params['exp_name'],
        exp_prefix=params['exp_prefix']
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('yaml', type=str)
    args = parser.parse_args()

    assert(os.path.exists(args.yaml))
    with open(args.yaml, 'r') as f:
        params.update(yaml.load(f))
    params['yaml_path'] = args.yaml

    main()
