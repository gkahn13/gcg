import numpy as np
import tensorflow as tf

from rllab.core.serializable import Serializable

from sandbox.gkahn.rnn_critic.policies.random_mac_policy import RandomMACPolicy

class FinalMACPolicy(RandomMACPolicy, Serializable):
    def __init__(self, **kwargs):
        Serializable.quick_init(self, locals())

        self._final_value_only = kwargs['final_value_only']

        RandomMACPolicy.__init__(self, **kwargs)

        assert(self._values_softmax['type'] == 'final')
        assert(self._get_action_test['values_softmax']['type'] == 'final')
        assert(self._get_action_target['values_softmax']['type'] == 'final')

    ###########################
    ### TF graph operations ###
    ###########################

    def _graph_inference(self, tf_obs_lowd, tf_actions_ph, values_softmax, tf_preprocess, add_reg=True):
        tf_values, tf_values_softmax, tf_nstep_rewards, tf_nstep_values = \
            super(RandomMACPolicy, self)._graph_inference(tf_obs_lowd, tf_actions_ph, values_softmax, tf_preprocess, add_reg)

        if self._final_value_only:
            tf_values = tf.concat(1, tf_nstep_values)
            batch_size = tf.shape(tf_obs_lowd)[0]
            tf_nstep_rewards = [tf.zeros((batch_size, 1)) for _ in range(len(tf_nstep_rewards))]

        return tf_values, tf_values_softmax, tf_nstep_rewards, tf_nstep_values

    ################
    ### Training ###
    ################

    def train_step(self, step, steps, observations, actions, rewards, values, dones, logprobs, use_target):
        """
        :param steps: [batch_size, N+1]
        :param observations: [batch_size, N+1 + obs_history_len-1, obs_dim]
        :param actions: [batch_size, N+1, action_dim]
        :param rewards: [batch_size, N+1]
        :param dones: [batch_size, N+1]
        """
        batch_size = len(steps)
        target_lens = self._N * np.ones((batch_size,), dtype=np.int32)

        feed_dict = {
            ### parameters
            self._tf_dict['lr_ph']: self._lr_schedule.value(step),
            self._tf_dict['explore_train_ph']: np.reshape([self._exploration_strategy.schedule.value(s)
                                                           for s in steps.ravel()], steps.shape),
            ### policy
            self._tf_dict['obs_ph']: observations[:, :self._obs_history_len, :],
            self._tf_dict['actions_ph']: actions,
            self._tf_dict['dones_ph']: np.logical_or(not use_target, dones[:, :self._N]),
            self._tf_dict['rewards_ph']: rewards[:, :self._N],
            self._tf_dict['logprob_ph']: logprobs[:, :self._N]
        }
        if self._use_target:
            # feed_dict[self._tf_dict['obs_target_ph']] = [observations[i, l - self._obs_history_len + 1:l + 1, :]
            #                                              for i, l in enumerate(target_lens)]
            feed_dict[self._tf_dict['obs_target_ph']] = [observations[i, l:l + self._obs_history_len, :]
                                                         for i, l in enumerate(target_lens)]
            feed_dict[self._tf_dict['target_len_ph']] = target_lens

        cost, mse, _ = self._tf_dict['sess'].run([self._tf_dict['cost'],
                                                  self._tf_dict['mse'],
                                                  self._tf_dict['opt']],
                                                 feed_dict=feed_dict)
        assert(np.isfinite(cost))

        self._log_stats['Cost'].append(cost)
        self._log_stats['mse/cost'].append(mse / cost)
