import os
import joblib
import numpy as np

from rllab.algos.base import RLAlgorithm
from rllab.misc.overrides import overrides
import rllab.misc.logger as logger

from sandbox.gkahn.rnn_critic.sampler.sampler import RNNCriticSampler
from sandbox.gkahn.rnn_critic.utils.utils import timeit

class RNNCritic(RLAlgorithm):

    def __init__(self,
                 env,
                 env_eval,
                 policy,
                 max_path_length,
                 total_steps,
                 sample_after_n_steps,
                 learn_after_n_steps,
                 train_every_n_steps,
                 eval_every_n_steps,
                 save_every_n_steps,
                 update_target_after_n_steps,
                 update_target_every_n_steps,
                 update_preprocess_every_n_steps,
                 log_every_n_steps,
                 batch_size,
                 onpolicy_after_n_steps=-1,
                 n_envs=1,
                 replay_pool_size=int(1e6),
                 replay_pool_sampling='uniform',
                 save_rollouts=False,
                 save_rollouts_observations=True,
                 save_eval_rollouts_observations=False,
                 offpolicy=None,
                 num_offpolicy=np.inf,
                 render=False,
                 env_str=None):
        assert(learn_after_n_steps % n_envs == 0)
        if train_every_n_steps >= 1:
            assert(int(train_every_n_steps) % n_envs == 0)
        assert(save_every_n_steps % n_envs == 0)
        assert(update_target_every_n_steps % n_envs == 0)
        assert(update_preprocess_every_n_steps % n_envs == 0)

        self._policy = policy
        self._max_path_length = int(max_path_length)
        self._total_steps = int(total_steps)
        self._sample_after_n_steps = int(sample_after_n_steps)
        self._onpolicy_after_n_steps = int(onpolicy_after_n_steps)
        self._learn_after_n_steps = int(learn_after_n_steps)
        self._train_every_n_steps = train_every_n_steps
        self._eval_every_n_steps = int(eval_every_n_steps)
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
            sampling_method=replay_pool_sampling,
            save_rollouts=save_rollouts,
            save_rollouts_observations=save_rollouts_observations,
            env_str=env_str
        )

        self._eval_sampler = RNNCriticSampler(
            policy=policy,
            env=env_eval,
            n_envs=1,
            replay_pool_size=int(np.ceil(1.5 * max_path_length) + 1),
            max_path_length=max_path_length,
            sampling_method=replay_pool_sampling,
            save_rollouts=True,
            save_rollouts_observations=save_eval_rollouts_observations
        )

        if offpolicy is not None:
            assert(os.path.exists(offpolicy))
            logger.log('Loading offpolicy data from {0}'.format(offpolicy))
            self._sampler.add_offpolicy(offpolicy, int(num_offpolicy))

    ####################
    ### Save methods ###
    ####################

    def _save_rollouts_file(self, itr, rollouts, eval=False):
        if eval:
            fname = 'itr_{0}_rollouts_eval.pkl'.format(itr)
        else:
            fname = 'itr_{0}_rollouts.pkl'.format(itr)
        fname = os.path.join(logger.get_snapshot_dir(), fname)
        joblib.dump({'rollouts': rollouts}, fname, compress=3)

    def _save_params(self, itr, train_rollouts, eval_rollouts):
        with self._policy.session.as_default(), self._policy.session.graph.as_default():
            itr_params = dict(
                itr=itr,
                policy=self._policy,
            )
            logger.save_itr_params(itr, itr_params)

            self._save_rollouts_file(itr, train_rollouts)
            self._save_rollouts_file(itr, eval_rollouts, eval=True)

    ########################
    ### Training methods ###
    ########################

    @overrides
    def train(self):
        save_itr = 0
        target_updated = False
        eval_rollouts = []

        timeit.reset()
        timeit.start('total')
        for step in range(0, self._total_steps, self._sampler.n_envs):
            ### sample and add to buffer
            if step > self._sample_after_n_steps:
                timeit.start('sample')
                self._sampler.step(step,
                                   take_random_actions=(step <= self._learn_after_n_steps or
                                                        step <= self._onpolicy_after_n_steps),
                                   explore=True)
                timeit.stop('sample')

            ### sample and DON'T add to buffer (for validation)
            if step % self._eval_every_n_steps == 0:
                # logger.log('Evaluating')
                timeit.start('eval')
                eval_rollouts_step = []
                eval_step = step
                while len(eval_rollouts_step) == 0:
                    self._eval_sampler.step(eval_step, explore=False)
                    eval_rollouts_step = self._eval_sampler.get_recent_paths()
                    eval_step += 1
                eval_rollouts += eval_rollouts_step
                timeit.stop('eval')

            if step > self._learn_after_n_steps:
                ### training step
                if self._train_every_n_steps >= 1:
                    if step % int(self._train_every_n_steps) == 0:
                        timeit.start('batch')
                        batch = self._sampler.sample(self._batch_size)
                        timeit.stop('batch')
                        timeit.start('train')
                        self._policy.train_step(step, *batch, use_target=target_updated)
                        timeit.stop('train')
                else:
                    for _ in range(int(1. / self._train_every_n_steps)):
                        timeit.start('batch')
                        batch = self._sampler.sample(self._batch_size)
                        timeit.stop('batch')
                        timeit.start('train')
                        self._policy.train_step(step, *batch, use_target=target_updated)
                        timeit.stop('train')

                ### update target network
                if step > self._update_target_after_n_steps and step % self._update_target_every_n_steps == 0:
                    # logger.log('Updating target network')
                    self._policy.update_target()
                    target_updated = True

                ### update preprocess
                if step % self._update_preprocess_every_n_steps == 0:
                    # logger.log('Updating preprocess')
                    self._policy.update_preprocess(self._sampler.statistics)

                ### log
                if step % self._log_every_n_steps == 0:
                    logger.log('step %.3e' % step)
                    logger.record_tabular('Step', step)
                    self._sampler.log()
                    self._eval_sampler.log(prefix='Eval')
                    self._policy.log()
                    logger.dump_tabular(with_prefix=False)
                    timeit.stop('total')
                    logger.log('\n'+str(timeit))
                    timeit.reset()
                    timeit.start('total')

            ### save model
            if step > 0 and step % self._save_every_n_steps == 0:
                logger.log('Saving files')
                self._save_params(save_itr,
                                  train_rollouts=self._sampler.get_recent_paths(),
                                  eval_rollouts=eval_rollouts)
                save_itr += 1
                eval_rollouts = []

        self._save_params(save_itr,
                          train_rollouts=self._sampler.get_recent_paths(),
                          eval_rollouts=eval_rollouts)
