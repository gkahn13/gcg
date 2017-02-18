import os

from sandbox.gkahn.rnn_critic.algos.rnn_critic import RNNCritic
from sandbox.gkahn.rnn_critic.policies.rnn_critic_mlp_policy import RNNCriticMLPPolicy
from sandbox.gkahn.rnn_critic.policies.rnn_critic_rnn_policy import RNNCriticRNNPolicy

from rllab.envs.gym_env import GymEnv
from sandbox.gkahn.rnn_critic.envs.point_env import PointEnv
from rllab.exploration_strategies.gaussian_strategy import GaussianStrategy
from rllab.envs.normalized_env import normalize
from rllab.misc.instrument import run_experiment_lite
from sandbox.rocky.tf.envs.base import TfEnv

import tensorflow as tf

def run_task(*_):
    # env = TfEnv(normalize(GymEnv("Reacher-v1")))
    env = TfEnv(normalize(PointEnv()))

    lattice_get_action_params = {
        'type': 'lattice',
        'N': 10
    }
    random_get_action_params = {
        'type': 'random',
        'N': 1000
    }

    rnn_critic_policy_params = dict(
        env_spec=env.spec,
        H=2,
        weight_decay=0.,
        learning_rate=0.001,
        reset_every_train=True,
        train_steps=1000,
        batch_size=16,
        get_action_params=random_get_action_params,
    )

    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3)
    config = tf.ConfigProto(gpu_options=gpu_options,
                            log_device_placement=False,
                            allow_soft_placement=True)
    sess = tf.Session(config=config)
    with sess.as_default():
        policy = RNNCriticMLPPolicy(
            hidden_layers=[40, 40],
            activation=tf.nn.relu,
            **rnn_critic_policy_params
        )
        # policy = RNNCriticRNNPolicy(
        #     obs_hidden_layers=[20],
        #     action_hidden_layers=[20],
        #     reward_hidden_layers=[10, 5],
        #     rnn_state_dim=20,
        #     activation=tf.nn.relu,
        #     **rnn_critic_policy_params
        # )

        algo = RNNCritic(
            env=env,
            policy=policy,
            n_rollouts=10000,
            max_path_length=env.horizon,
            exploration_strategy=GaussianStrategy(env.spec, max_sigma=0.5, min_sigma=0.01),
            train_every_n_rollouts=10, # 1000
            render=False,
            store_rollouts=True
        )
        algo.train()

seed = 1
tf.set_random_seed(seed)

# run_task()

run_experiment_lite(
    run_task,
    # Number of parallel workers for sampling
    n_parallel=2,
    # Only keep the snapshot parameters for the last iteration
    snapshot_mode="all",
    # Specifies the seed for the experiment. If this is not provided, a random seed
    # will be used
    seed=seed,
    # plot=True,
    exp_name='test',
    exp_prefix='rnn_critic'
)
