import inspect
import time

import numpy as np

from rllab.algos.base import RLAlgorithm
from rllab.misc.overrides import overrides
import rllab.misc.logger as logger
from sandbox.gkahn.rnn_critic.algos import rnn_util
from sandbox.gkahn.rnn_critic.sampler.rnn_critic_batch_sampler import RNNCriticBatchSampler
from sandbox.gkahn.rnn_critic.algos.rnn_critic_sampler import RNNCriticSampler

class RNNCritic(RLAlgorithm):

    def __init__(self,
                 env,
                 sampling_policy,
                 training_policy,
                 n_rollouts,
                 max_path_length,
                 exploration_strategy,
                 train_every_n_rollouts,
                 render=False,
                 is_async=False):
        """
        :param env: Environment
        :param sampling_policy: RNNCriticPolicy
        :param training_policy: RNNCriticPolicy
        :param n_rollouts: maximum number of rollouts to train with
        :param max_path_length: maximum length of a single rollout
        :param exploration_strategy: how actions are modified
        :param train_every_n_rollouts: train policy every __ rollouts
        :param render: show env
        :param is_async: asynchronous sampling/training
        """
        self._env = env
        self._sampling_policy = sampling_policy
        self._training_policy = training_policy
        self._n_rollouts = n_rollouts
        self._max_path_length = max_path_length
        self._train_every_n_rollouts = train_every_n_rollouts
        self._is_async = is_async

        # for attr in [attr for attr in dir(self) if '__' not in attr and not inspect.ismethod(getattr(self, attr))]:
        #     logger.log('RNNCritic\t{0}: {1}'.format(attr, getattr(self, attr)))

        sampling_policy.set_exploration_strategy(exploration_strategy)

        self._sampler = RNNCriticSampler(
            env=env,
            policy=sampling_policy,
            rollouts_per_sample=self._train_every_n_rollouts,
            max_path_length=self._max_path_length,
            render=render
        )

    @overrides
    def train(self):
        assert(not self._is_async)
        self._sync_train()

    def _sync_train(self):
        for itr in range(self._n_rollouts // self._train_every_n_rollouts):
            with logger.prefix('itr #{0:d} |'.format(itr)):
                ### sample rollouts
                logger.log('Sampling rollouts...')
                self._sampler.sample_rollouts()
                tfrecords, preprocess_stats, sampler_log = self._sampler.get_tfrecords_and_statistics()
                ### train
                logger.log('Training policy...')
                train_log = self._training_policy.train(tfrecords, preprocess_stats)
                self._sampler.update_policy(self._training_policy)

                ### save pkl
                with self._sampling_policy.session.as_default():
                    itr_params = dict(
                        itr=itr,
                        policy=self._sampling_policy,
                        env=self._env,
                        train_log=train_log
                    )
                    logger.save_itr_params(itr, itr_params)

                ### log
                for k in ('FinalRewardMean', 'FinalRewardStd', 'AvgRewardMean', 'AvgRewardStd', 'RolloutTime'):
                    logger.record_tabular(k, sampler_log[k])
                logger.record_tabular('InitialTrainCost', train_log['cost'][0])
                logger.record_tabular('FinalTrainCost', train_log['cost'][-1])
                logger.record_tabular('TrainTime', train_log['TrainTime'])
                logger.dump_tabular(with_prefix=False)


class RNNCriticOLD(RLAlgorithm):

    def __init__(self,
                 env,
                 policy,
                 n_rollouts,
                 max_path_length,
                 exploration_strategy,
                 train_every_n_rollouts,
                 render=False,
                 store_rollouts=False):
        """
        :param env: Environment
        :param policy: RNNCriticPolicy
        :param n_rollouts: maximum number of rollouts to train with
        :param max_path_length: maximum length of a single rollout
        :param exploration_strategy: how actions are modified
        :param train_every_n_rollouts: train policy every __ rollouts
        :param render: show env
        """
        self._env = env
        self._policy = policy
        self._n_rollouts = n_rollouts
        self._max_path_length = max_path_length
        self._train_every_n_rollouts = train_every_n_rollouts
        self._exploration_strategy = exploration_strategy
        self._render = render
        self._store_rollouts = store_rollouts

        # for attr in [attr for attr in dir(self) if '__' not in attr and not inspect.ismethod(getattr(self, attr))]:
        #     logger.log('RNNCritic\t{0}: {1}'.format(attr, getattr(self, attr)))

        policy.set_exploration_strategy(exploration_strategy)

        self._sampler = RNNCriticBatchSampler(
            env=env,
            policy=policy,
            max_path_length=max_path_length,
            batch_size=train_every_n_rollouts * max_path_length,
            scope=None
        )

    @overrides
    def train(self):
        self._sampler.start_worker()

        replay_pool = rnn_util.RNNCriticReplayPool()

        for itr in range(self._n_rollouts // self._train_every_n_rollouts):
            itr_start_time = time.time()
            with logger.prefix('itr #{0:d} |'.format(itr)):
                logger.log('Obtaining samples...')
                paths = self._sampler.obtain_samples(itr)

                logger.log('Processing samples...')
                rollouts = self._sampler.process_samples(itr, paths)
                for rollout in rollouts[:self._train_every_n_rollouts]:
                    replay_pool.add_rollout(rollout)

                logger.log('Optimizing policy...')
                train_log = self._policy.train(replay_pool)

                logger.log('Saving snapshot...')
                itr_params = dict(
                    itr=itr,
                    policy=self._policy,
                    env=self._env,
                    train_log=train_log
                )
                if self._store_rollouts:
                    itr_params['rollouts'] = replay_pool.get_rollouts(last_n_rollouts=self._train_every_n_rollouts)
                logger.save_itr_params(itr, itr_params)

                rollout_stats = replay_pool.statistics(last_n_rollouts=self._train_every_n_rollouts)
                logger.record_tabular('InitialTrainCost', train_log['cost'][0])
                logger.record_tabular('FinalTrainCost', train_log['cost'][-1])
                logger.record_tabular('FinalRewardMean', np.mean(rollout_stats['final_reward']))
                logger.record_tabular('FinalRewardStd', np.std(rollout_stats['final_reward']))
                logger.record_tabular('AvgRewardMean', np.mean(rollout_stats['avg_reward']))
                logger.record_tabular('AvgRewardStd', np.std(rollout_stats['avg_reward']))
                logger.record_tabular('ItrTime', time.time() - itr_start_time)
                logger.dump_tabular(with_prefix=False)

        self._sampler.shutdown_worker()
