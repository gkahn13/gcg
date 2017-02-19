import inspect

from rllab.algos.base import RLAlgorithm
from rllab.misc.overrides import overrides
import rllab.misc.logger as logger
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
                 n_envs=1,
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
            n_envs=n_envs,
            render=render
        )

    @overrides
    def train(self):
        assert(not self._is_async)
        self._sync_train()

    def _sync_train(self):
        for itr in range(self._n_rollouts // self._train_every_n_rollouts):
            with logger.prefix('itr #{0:d} | '.format(itr)):
                ### sample rollouts
                logger.log('Sampling rollouts...')
                self._sampler.sample_rollouts()
                tfrecords, paths, preprocess_stats, sampler_log = self._sampler.get_tfrecords_paths_and_statistics()
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
                        train_log=train_log,
                        rollouts=paths
                    )
                    logger.save_itr_params(itr, itr_params)

                ### log
                keys = ('FinalRewardMean', 'FinalRewardStd', 'AvgRewardMean', 'AvgRewardStd', 'RolloutTime')
                for k in keys:
                    logger.record_tabular(k, sampler_log[k])
                for k in [k for k in sampler_log.keys() if k not in keys]:
                    logger.record_tabular(k, sampler_log[k])
                logger.record_tabular('InitialTrainCost', train_log['cost'][0])
                logger.record_tabular('FinalTrainCost', train_log['cost'][-1])
                logger.record_tabular('TrainTime', train_log['TrainTime'])
                logger.dump_tabular(with_prefix=False)
