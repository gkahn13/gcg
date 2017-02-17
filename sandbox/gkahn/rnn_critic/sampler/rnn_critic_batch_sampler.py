from rllab.sampler.base import Sampler
from rllab.sampler import parallel_sampler
from rllab.sampler.stateful_pool import singleton_pool
import tensorflow as tf

from sandbox.gkahn.rnn_critic.algos.rnn_util import Rollout

def worker_init_tf(G):
    G.sess = tf.Session()
    G.sess.__enter__()

def worker_init_tf_vars(G):
    G.sess.run(tf.initialize_all_variables())

class RNNCriticBatchSampler(Sampler):
    def __init__(self, env, policy, max_path_length, batch_size, scope=None):
        self._env = env
        self._policy = policy
        self._max_path_length = max_path_length
        self._batch_size = batch_size
        self._scope = scope

    def start_worker(self):
        if singleton_pool.n_parallel > 1:
            singleton_pool.run_each(worker_init_tf)
        parallel_sampler.populate_task(self._env, self._policy)
        # if singleton_pool.n_parallel > 1: # TODO: don't think I need to init tf vars since policy does that
        #     singleton_pool.run_each(worker_init_tf_vars)

    def obtain_samples(self, itr):
        cur_policy_params = self._policy.get_param_values()
        if hasattr(self._env, 'get_param_values'):
            cur_env_params = self._env.get_param_values()
        else:
            cur_env_params = None

        paths = parallel_sampler.sample_paths(
            policy_params=cur_policy_params,
            env_params=cur_env_params,
            max_samples=self._batch_size,
            max_path_length=self._max_path_length,
            scope=self._scope,
        )

        return paths

    def process_samples(self, itr, paths):
        rollouts = []
        for path in paths:
            rollout = Rollout()
            rollout.observations = path['observations']
            rollout.actions = path['actions']
            rollout.rewards = path['rewards']
            rollout.terminals = [False] * len(path['observations'])
            rollouts.append(rollout)

        return rollouts

    def shutdown_worker(self):
        parallel_sampler.terminate_task(scope=self._scope)

