import os
import joblib
import pickle
import torch

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
                 save_rollouts_observations=True,
                 offpolicy=None,
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
        :param save_rollouts_observations: if save_rollouts, save the observations?
        :param offpolicy: params about doing offpolicy
        :param render: show env
        """
        assert(learn_after_n_steps % n_envs == 0)
        if train_every_n_steps >= 1:
            assert(int(train_every_n_steps) % n_envs == 0)
        assert(save_every_n_steps % n_envs == 0)
        assert(update_target_every_n_steps % n_envs == 0)
        assert(update_preprocess_every_n_steps % n_envs == 0)

        self._env = env
        self._policy = policy
        self._max_path_length = int(max_path_length)
        self._total_steps = int(total_steps)
        self._learn_after_n_steps = int(learn_after_n_steps)
        self._train_every_n_steps = train_every_n_steps
        self._save_every_n_steps = int(save_every_n_steps)
        self._update_target_after_n_steps = int(update_target_after_n_steps)
        self._update_target_every_n_steps = int(update_target_every_n_steps)
        self._update_preprocess_every_n_steps = int(update_preprocess_every_n_steps)
        self._log_every_n_steps = int(log_every_n_steps)
        self._batch_size = batch_size
        self._save_rollouts = save_rollouts
        self._save_rollouts_observations = save_rollouts_observations

        self._sampler = RNNCriticSampler(
            policy=policy,
            env=env,
            n_envs=n_envs,
            replay_pool_size=replay_pool_size,
            max_path_length=max_path_length,
            save_rollouts=save_rollouts,
            save_rollouts_observations=save_rollouts_observations
        )

        if offpolicy is not None:
            logger.log('Loading offpolicy data from {0}'.format(offpolicy['folder']))
            self._sampler.add_offpolicy(offpolicy['folder'])

            logger.log('Training with offpolicy data')
            eval_sampler = RNNCriticSampler(
                policy=policy,
                env=env,
                n_envs=1,
                replay_pool_size=env.horizon,
                max_path_length=max_path_length,
                save_rollouts=False,
                save_rollouts_observations=False
            )
            self._train_offpolicy(offpolicy, eval_sampler)

        ### set exploration after offpolicy
        policy.set_exploration_strategy(exploration_strategy)

    ####################
    ### File methods ###
    ####################

    def _params_file(self, itr):
        return os.path.join(logger.get_snapshot_dir(), 'itr_{0}.pth.tar'.format(itr))

    def _rollouts_file(self, itr):
        return os.path.join(logger.get_snapshot_dir(), 'itr_{0}_rollouts.pkl'.format(itr))

    ########################
    ### Training methods ###
    ########################

    def _save_params(self, itr):
        env_is_pickleable = True
        try:
            pickle.dumps(self._env)
        except:
            env_is_pickleable=False

        itr_params = dict(
            itr=itr,
            policy=self._policy,
            policy_state_dicts=self._policy.state_dicts()
            # env=self._env if env_is_pickleable else None
        )
        torch.save(itr_params, self._params_file(itr))

        joblib.dump({'rollouts': self._sampler.get_recent_paths()},
                    self._rollouts_file(itr), compress=3)

    def _train_offpolicy(self, offpolicy, eval_sampler):
        total_steps = int(offpolicy['total_steps'])
        update_target_after_n_steps = int(offpolicy['update_target_after_n_steps'])
        update_target_every_n_steps = int(offpolicy['update_target_every_n_steps'])
        n_evals_per_step = int(offpolicy['n_evals_per_step'])
        log_every_n_steps = int(offpolicy['log_every_n_steps'])

        ### update preprocess
        stats = self._sampler.statistics
        self._policy.update_preprocess(stats)

        target_updated = False
        eval_sampler_step = 0
        for step in range(total_steps):
            for _ in range(n_evals_per_step):
                eval_sampler.step(eval_sampler_step)
                eval_sampler_step += 1

            ### train
            batch = self._sampler.sample(self._batch_size)
            self._policy.train_step(step,
                                    *batch,
                                    use_target=target_updated)

            ### target
            if step > update_target_after_n_steps and \
               step % update_target_every_n_steps == 0:
                self._policy.update_target()
                target_updated = True

            ### log
            if step % log_every_n_steps == 0:
                logger.log('offpolicy step %.3e' % step)
                logger.record_tabular('Step', step)
                eval_sampler.log()
                self._policy.log()
                logger.dump_tabular(with_prefix=False)

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
                    if step % int(self._train_every_n_steps) == 0:
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
                        timeit.start('batch')
                        batch = self._sampler.sample(self._batch_size)
                        timeit.stop('batch')
                        timeit.start('train')
                        self._policy.train_step(step,
                                                *batch,
                                                use_target=target_updated)
                        timeit.stop('train')

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
            if step > 0 and step % self._save_every_n_steps == 0:
                # logger.log('Saving...')
                self._save_params(save_itr)
                save_itr += 1

        self._save_params(save_itr)

