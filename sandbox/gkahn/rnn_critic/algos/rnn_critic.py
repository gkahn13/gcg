import pickle

from rllab.algos.base import RLAlgorithm
from rllab.misc.overrides import overrides
import rllab.misc.logger as logger
from sandbox.gkahn.rnn_critic.sampler.sampler import RNNCriticSampler
from sandbox.gkahn.rnn_critic.utils.utils import timeit

class RNNCritic(RLAlgorithm):

    def __init__(self,
                 env,
                 policy,
                 max_path_length,
                 exploration_strategy,
                 total_steps,
                 learn_after_n_steps,
                 train_every_n_steps,
                 save_every_n_steps,
                 update_target_after_n_steps,
                 update_target_every_n_steps,
                 update_preprocess_every_n_steps,
                 log_every_n_steps,
                 batch_size,
                 n_envs=1,
                 replay_pool_size=int(1e6),
                 save_rollouts=False,
                 render=False):
        """
        :param env: Environment
        :param policy: RNNCriticPolicy
        :param max_path_length: maximum length of a single rollout
        :param exploration_strategy: how actions are modified
        :param total_steps: how many steps to take in total
        :param learn_after_n_steps: before this many steps, take random actions
        :param train_every_n_steps: train policy every n steps
        :param save_every_n_steps: save policy every n steps
        :param update_target_after_n_steps:
        :param update_target_every_n_steps: update target every n steps
        :param update_preprocess_every_n_steps
        :param log_every_n_steps: log every n steps
        :param batch_size: batch size per gradient step
        :param save_rollouts: save rollouts when saving params?
        :param render: show env
        """
        assert(learn_after_n_steps % n_envs == 0)
        if train_every_n_steps >= 1:
            assert(train_every_n_steps % n_envs == 0)
        assert(save_every_n_steps % n_envs == 0)
        assert(update_target_every_n_steps % n_envs == 0)
        assert(update_preprocess_every_n_steps % n_envs == 0)

        self._env = env
        self._policy = policy
        self._max_path_length = int(max_path_length)
        self._total_steps = int(total_steps)
        self._learn_after_n_steps = int(learn_after_n_steps)
        self._train_every_n_steps = int(train_every_n_steps)
        self._save_every_n_steps = int(save_every_n_steps)
        self._update_target_after_n_steps = int(update_target_after_n_steps)
        self._update_target_every_n_steps = int(update_target_every_n_steps)
        self._update_preprocess_every_n_steps = int(update_preprocess_every_n_steps)
        self._log_every_n_steps = int(log_every_n_steps)
        self._batch_size = batch_size
        self._save_rollouts = save_rollouts

        policy.set_exploration_strategy(exploration_strategy)

        self._sampler = RNNCriticSampler(
            policy=policy,
            env=env,
            n_envs=n_envs,
            replay_pool_size=replay_pool_size,
            max_path_length=max_path_length,
            save_rollouts=save_rollouts
        )

    def _save_params(self, itr):
        env_is_pickleable = True
        try:
            pickle.dumps(self._env)
        except:
            env_is_pickleable=False

        with self._policy.session.as_default(), self._policy.session.graph.as_default():
            itr_params = dict(
                itr=itr,
                policy=self._policy,
                env=self._env if env_is_pickleable else None,
                rollouts=self._sampler.get_recent_paths()
            )
            logger.save_itr_params(itr, itr_params)

    @overrides
    def train(self):
        save_itr = 0
        target_updated = False
        timeit.reset()
        timeit.start('total')
        for step in range(0, self._total_steps, self._sampler.n_envs):
            timeit.start('sample')
            self._sampler.step(step, take_random_actions=(step <= self._learn_after_n_steps))
            timeit.stop('sample')

            if step > self._learn_after_n_steps:
                ### training step
                if self._train_every_n_steps >= 1:
                    if step % self._train_every_n_steps == 0:
                        timeit.start('batch')
                        batch = self._sampler.sample(self._batch_size)
                        timeit.stop('batch')
                        timeit.start('train')
                        self._policy.train_step(step,
                                                *batch,
                                                use_target=target_updated)
                        timeit.stop('train')
                else:
                    for _ in range(int(1. / self._train_every_n_steps)):
                        self._policy.train_step(step,
                                                *self._sampler.sample(self._batch_size),
                                                use_target=target_updated)

                ### update target network
                if step > self._update_target_after_n_steps and step % self._update_target_every_n_steps == 0:
                    # logger.log('Updating target network...')
                    timeit.start('target')
                    self._policy.update_target()
                    target_updated = True
                    timeit.stop('target')

                ### update preprocess
                if step % self._update_preprocess_every_n_steps == 0:
                    timeit.start('stats')
                    stats = self._sampler.statistics
                    timeit.stop('stats')
                    timeit.start('preprocess')
                    self._policy.update_preprocess(stats)
                    timeit.stop('preprocess')

                if step % self._log_every_n_steps == 0:
                    logger.log('step %.3e' % step)
                    logger.record_tabular('Step', step)
                    self._sampler.log()
                    self._policy.log()
                    logger.dump_tabular(with_prefix=False)
                    timeit.stop('total')
                    print('')
                    print(str(timeit))
                    print('')
                    timeit.reset()
                    timeit.start('total')

                ### save model
                if step % self._save_every_n_steps == 0:
                    # logger.log('Saving...')
                    self._save_params(save_itr)
                    save_itr += 1

        self._save_params(save_itr)

