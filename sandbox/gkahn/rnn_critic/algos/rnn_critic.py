import inspect

from rllab.algos.base import RLAlgorithm
from rllab.misc.overrides import overrides
import rllab.misc.logger as logger
from sandbox.gkahn.rnn_critic.sampler.sampler import RNNCriticSampler

class RNNCritic(RLAlgorithm):

    def __init__(self,
                 env,
                 policy,
                 n_rollouts,
                 max_path_length,
                 exploration_strategy,
                 total_steps,
                 learn_after_n_steps,
                 train_every_n_steps,
                 save_every_n_steps,
                 update_target_every_n_steps,
                 log_every_n_steps,
                 batch_size,
                 n_envs=1,
                 replay_pool_size=int(1e6),
                 render=False):
        """
        :param env: Environment
        :param policy: RNNCriticPolicy
        :param n_rollouts: maximum number of rollouts to train with
        :param max_path_length: maximum length of a single rollout
        :param exploration_strategy: how actions are modified
        :param total_steps: how many steps to take in total
        :param learn_after_n_steps: before this many steps, take random actions
        :param train_every_n_steps: train policy every n steps
        :param save_every_n_steps: save policy every n steps
        :param update_target_every_n_steps: update target every n steps
        :param log_every_n_steps: log every n steps
        :param batch_size: batch size per gradient step
        :param render: show env
        :param is_async: asynchronous sampling/training
        """
        assert(learn_after_n_steps % n_envs == 0)
        assert(train_every_n_steps % n_envs == 0)
        assert(save_every_n_steps % n_envs == 0)
        assert(update_target_every_n_steps % n_envs == 0)

        self._env = env
        self._policy = policy
        self._n_rollouts = n_rollouts
        self._max_path_length = max_path_length
        self._total_steps = total_steps
        self._learn_after_n_steps = learn_after_n_steps
        self._train_every_n_steps = train_every_n_steps
        self._save_every_n_steps = save_every_n_steps
        self._update_target_every_n_steps = update_target_every_n_steps
        self._log_every_n_steps = log_every_n_steps
        self._batch_size = batch_size

        # for attr in [attr for attr in dir(self) if '__' not in attr and not inspect.ismethod(getattr(self, attr))]:
        #     logger.log('RNNCritic\t{0}: {1}'.format(attr, getattr(self, attr)))

        policy.set_exploration_strategy(exploration_strategy)

        self._sampler = RNNCriticSampler(
            policy=policy,
            env=env,
            n_envs=n_envs,
            replay_pool_size=replay_pool_size,
            max_path_length=max_path_length
        )

    @overrides
    def train(self):
        save_itr = 0
        for step in range(0, self._total_steps, self._sampler.n_envs):
            self._sampler.step()

            if step > self._learn_after_n_steps:
                ### training step
                if step % self._train_every_n_steps == 0:
                    self._policy.train_step(*self._sampler.sample(self._batch_size))

                ### update target network
                if step % self._update_target_every_n_steps == 0:
                    pass # TODO

                if step % self._log_every_n_steps == 0:
                    self._sampler.log()
                    self._policy.log()
                    logger.dump_tabular(with_prefix=False)

                ### save model
                if step % self._save_every_n_steps == 0:
                    with self._policy.session.as_default():
                        itr_params = dict(
                            itr=save_itr,
                            policy=self._policy,
                            env=self._env,
                            rollouts=self._sampler.get_recent_paths()
                        )
                        logger.save_itr_params(save_itr, itr_params)
                    save_itr += 1



    @overrides
    def train_OLD(self):
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
