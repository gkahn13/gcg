import os
import joblib
import pickle

import rllab.misc.logger as logger
from rllab.misc.overrides import overrides
from rllab.algos.base import RLAlgorithm

from sandbox.gkahn.rnn_critic.sampler.replay_pool import RNNCriticReplayPool
from sandbox.gkahn.rnn_critic.policies.mac_policy import MACPolicy

class RNNCriticOffpolicy(RLAlgorithm):

    def __init__(self,
                 env,
                 policy,
                 offpolicy_data,
                 total_steps,
                 save_every_n_steps,
                 update_target_after_n_steps,
                 update_target_every_n_steps,
                 log_every_n_steps,
                 batch_size,
                 replay_pool_size=int(1e6),
                 **kwargs):
        """
        :param env: Environment
        :param policy: RNNCriticPolicy
        :param offpolicy_data: absolute path to directory containing itr_${i}.pkl files with rollouts
        :param total_steps: how many steps to take in total
        :param save_every_n_steps: save policy every n steps
        :param update_target_after_n_steps:
        :param update_target_every_n_steps: update target every n steps
        :param log_every_n_steps: log every n steps
        :param batch_size: batch size per gradient step
        """
        self._env = env
        self._policy = policy
        self._offpolicy_data = offpolicy_data
        self._total_steps = int(total_steps)
        self._save_every_n_steps = int(save_every_n_steps)
        self._update_target_after_n_steps = int(update_target_after_n_steps)
        self._update_target_every_n_steps = int(update_target_every_n_steps)
        self._log_every_n_steps = int(log_every_n_steps)
        self._batch_size = batch_size

        self._replay_pool = RNNCriticReplayPool(env.spec,
                                                policy.N,
                                                replay_pool_size,
                                                obs_history_len=policy.obs_history_len,
                                                save_rollouts=False)
        self._fill_replay_pool()

    ##############################
    ### Offpolicy data methods ###
    ##############################

    def _offpolicy_itr_file(self, itr):
        return os.path.join(self._offpolicy_data, 'itr_{0:d}.pkl'.format(itr))

    def _load_rollouts(self):
        rollouts = []
        itr = 0

        while os.path.exists(self._offpolicy_itr_file(itr)):
            sess, graph = MACPolicy.create_session_and_graph(gpu_device=0, gpu_frac=0.1)
            with graph.as_default(), sess.as_default():
                d = joblib.load(self._offpolicy_itr_file(itr))
                rollouts += d['rollouts']
                d['policy'].terminate()
            itr += 1

        return rollouts

    def _fill_replay_pool(self):
        rollouts = self._load_rollouts()
        step = 0
        for rollout in rollouts:
            for obs, action, reward, done in zip(rollout['observations'],
                                                 rollout['actions'],
                                                 rollout['rewards'],
                                                 rollout['dones']):
                self._replay_pool.store_observation(step, obs)
                self._replay_pool.store_effect(action, reward, done, flatten_action=False)
                step += 1

    ########################
    ### Training methods ###
    ########################

    def _save_params(self, itr):
        env_is_pickleable = True
        try:
            pickle.dumps(self._env)
        except:
            env_is_pickleable = False

        with self._policy.session.as_default(), self._policy.session.graph.as_default():
            itr_params = dict(
                itr=itr,
                policy=self._policy,
                env=self._env if env_is_pickleable else None,
                rollouts=[]
            )
            logger.save_itr_params(itr, itr_params)

    @overrides
    def train(self):
        ### no new data, so update it once at beginning
        self._policy.update_preprocess(self._replay_pool.statistics)

        save_itr = 0
        target_updated = False
        for step in range(0, self._total_steps):
            batch = self._replay_pool.sample(self._batch_size)
            self._policy.train_step(step,
                                    *batch,
                                    use_target=target_updated)

            ### update target network
            if step > self._update_target_after_n_steps and step % self._update_target_every_n_steps == 0:
                self._policy.update_target()
                target_updated = True

            if step % self._log_every_n_steps == 0:
                logger.log('step %.3e' % step)
                logger.record_tabular('Step', step)
                self._policy.log()
                logger.dump_tabular(with_prefix=False)

            ### save model
            if step > 0 and step % self._save_every_n_steps == 0:
                self._save_params(save_itr)
                save_itr += 1

        self._save_params(save_itr)

