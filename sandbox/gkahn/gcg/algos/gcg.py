import os
import joblib
import numpy as np

from rllab.algos.base import RLAlgorithm
from rllab.misc.overrides import overrides
import rllab.misc.logger as logger
from rllab import config

from sandbox.gkahn.gcg.envs.env_utils import create_env
from sandbox.gkahn.gcg.policies.mac_policy import MACPolicy
from sandbox.gkahn.gcg.policies.rccar_mac_policy import RCcarMACPolicy
from sandbox.gkahn.gcg.sampler.sampler import RNNCriticSampler
from sandbox.gkahn.gcg.utils.utils import timeit

class GCG(RLAlgorithm):

    def __init__(self, **kwargs):

        self._policy = kwargs['policy']

        self._batch_size = kwargs['batch_size']
        self._save_rollouts = kwargs['save_rollouts']
        self._save_rollouts_observations = kwargs['save_rollouts_observations']

        self._sampler = RNNCriticSampler(
            policy=kwargs['policy'],
            env=kwargs['env'],
            n_envs=kwargs['n_envs'],
            replay_pool_size=kwargs['replay_pool_size'],
            max_path_length=kwargs['max_path_length'],
            sampling_method=kwargs['replay_pool_sampling'],
            save_rollouts=kwargs['save_rollouts'],
            save_rollouts_observations=kwargs['save_rollouts_observations'],
            save_env_infos=kwargs['save_env_infos'],
            env_str=kwargs['env_str'],
            replay_pool_params=kwargs['replay_pool_params']
        )

        self._eval_sampler = RNNCriticSampler(
            policy=kwargs['policy'],
            env=kwargs['env_eval'],
            n_envs=1,
            replay_pool_size=int(np.ceil(1.5 * kwargs['max_path_length']) + 1),
            max_path_length=kwargs['max_path_length'],
            sampling_method=kwargs['replay_pool_sampling'],
            save_rollouts=True,
            save_rollouts_observations=kwargs.get('save_eval_rollouts_observations', False),
            save_env_infos=kwargs['save_env_infos'],
            replay_pool_params=kwargs['replay_pool_params']
        )

        if kwargs.get('offpolicy', None) is not None:
            assert(os.path.exists(kwargs['offpolicy']))
            logger.log('Loading offpolicy data from {0}'.format(kwargs['offpolicy']))
            self._sampler.add_offpolicy(kwargs['offpolicy'], int(kwargs['num_offpolicy']))
            logger.log('Added {0} samples'.format(len(self._sampler)))

        alg_args = kwargs
        self._total_steps = int(alg_args['total_steps'])
        self._sample_after_n_steps = int(alg_args['sample_after_n_steps'])
        self._onpolicy_after_n_steps = int(alg_args['onpolicy_after_n_steps'])
        self._learn_after_n_steps = int(alg_args['learn_after_n_steps'])
        self._train_every_n_steps = alg_args['train_every_n_steps']
        self._eval_every_n_steps = int(alg_args['eval_every_n_steps'])
        self._save_every_n_steps = int(alg_args['save_every_n_steps'])
        self._update_target_after_n_steps = int(alg_args['update_target_after_n_steps'])
        self._update_target_every_n_steps = int(alg_args['update_target_every_n_steps'])
        self._update_preprocess_every_n_steps = int(alg_args['update_preprocess_every_n_steps'])
        self._log_every_n_steps = int(alg_args['log_every_n_steps'])
        assert (self._learn_after_n_steps % self._sampler.n_envs == 0)
        if self._train_every_n_steps >= 1:
            assert (int(self._train_every_n_steps) % self._sampler.n_envs == 0)
        assert (self._save_every_n_steps % self._sampler.n_envs == 0)
        assert (self._update_target_every_n_steps % self._sampler.n_envs == 0)
        assert (self._update_preprocess_every_n_steps % self._sampler.n_envs == 0)

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
            if step > 0 and step % self._eval_every_n_steps == 0:
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

            if step >= self._learn_after_n_steps:
                ### update preprocess
                if step == self._learn_after_n_steps or step % self._update_preprocess_every_n_steps == 0:
                    # logger.log('Updating preprocess')
                    self._policy.update_preprocess(self._sampler.statistics)

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

def run_gcg(params):
    # copy yaml for posterity
    try:
        yaml_path = os.path.join(logger.get_snapshot_dir(), '{0}.yaml'.format(params['exp_name']))
        with open(yaml_path, 'w') as f:
            f.write(params['txt'])
    except:
        pass

    os.environ["CUDA_VISIBLE_DEVICES"] = str(params['policy']['gpu_device'])  # TODO: hack so don't double GPU
    config.USE_TF = True

    normalize_env = params['alg'].pop('normalize_env')

    env_str = params['alg'].pop('env')
    env = create_env(env_str, is_normalize=normalize_env, seed=params['seed'])

    env_eval_str = params['alg'].pop('env_eval', env_str)
    env_eval = create_env(env_eval_str, is_normalize=normalize_env, seed=params['seed'])

    env.reset()
    env_eval.reset()

    #####################
    ### Create policy ###
    #####################

    policy_class = params['policy']['class']
    PolicyClass = eval(policy_class)
    policy_params = params['policy'][policy_class]

    policy = PolicyClass(
        env_spec=env.spec,
        exploration_strategies=params['alg'].pop('exploration_strategies'),
        **policy_params,
        **params['policy']
    )

    ########################
    ### Create algorithm ###
    ########################

    if 'max_path_length' in params['alg']:
        max_path_length = params['alg'].pop('max_path_length')
    else:
        max_path_length = env.horizon
    algo = GCG(
        env=env,
        env_eval=env_eval,
        policy=policy,
        max_path_length=max_path_length,
        env_str=env_str,
        **params['alg']
    )
    algo.train()