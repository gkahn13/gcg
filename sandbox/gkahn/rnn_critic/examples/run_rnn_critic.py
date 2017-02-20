import os, argparse, yaml
import tensorflow as tf

from rllab.misc.instrument import run_experiment_lite
### environments
from sandbox.rocky.tf.envs.base import TfEnv
from rllab.envs.normalized_env import normalize
### exploration strategies
from rllab.exploration_strategies.gaussian_strategy import GaussianStrategy
### RNN critic
from sandbox.gkahn.rnn_critic.algos.rnn_critic import RNNCritic
from sandbox.gkahn.rnn_critic.policies.rnn_critic_mlp_policy import RNNCriticMLPPolicy
from sandbox.gkahn.rnn_critic.policies.rnn_critic_rnn_policy import RNNCriticRNNPolicy

### parameters loaded from yaml
params = dict()

def run_task(*_):
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

    sampling_policy = create_policy(is_train=False)
    training_policy = create_policy(is_train=True)

    es_params = params['alg'].pop('exploration_strategy')
    es_type = es_params['type']
    if es_type == 'gaussian':
        ESClass = GaussianStrategy
    else:
        raise Exception('Exploration strategy {0} not valid'.format(es_type))

    exploration_strategy = ESClass(env_spec=env.spec, **es_params[es_type])

    algo = RNNCritic(
        env=env,
        sampling_policy=sampling_policy,
        training_policy=training_policy,
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

    main()
