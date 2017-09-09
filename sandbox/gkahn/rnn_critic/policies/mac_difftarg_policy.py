import numpy as np
import tensorflow as tf

from rllab.core.serializable import Serializable
from rllab.misc import ext

from sandbox.gkahn.rnn_critic.policies.mac_policy import MACPolicy
from sandbox.gkahn.tf.core import xplatform

class MACDifftargPolicy(MACPolicy, Serializable):
    def __init__(self, **kwargs):
        Serializable.quick_init(self, locals())

        MACPolicy.__init__(self, **kwargs)

        assert(self._use_target)
        assert(self._separate_target_params)
        assert(self._get_action_test['H'] == self._H)
        assert(self._get_action_target['H'] < self._H)

    ###########################
    ### TF graph operations ###
    ###########################

    def _graph_setup(self):
        ### create session and graph
        tf_sess = tf.get_default_session()
        if tf_sess is None:
            tf_sess, tf_graph = MACPolicy.create_session_and_graph(gpu_device=self._gpu_device, gpu_frac=self._gpu_frac)
        tf_graph = tf_sess.graph

        with tf_sess.as_default(), tf_graph.as_default():
            if ext.get_seed() is not None:
                ext.set_seed(ext.get_seed())

            ### create input output placeholders
            tf_obs_ph, tf_actions_ph, tf_dones_ph, tf_rewards_ph, tf_obs_target_ph, \
            tf_test_es_ph_dict, tf_episode_timesteps_ph = self._graph_input_output_placeholders()
            self.global_step = tf.Variable(0, trainable=False, name='global_step')

            ##############
            ### policy ###
            ##############

            policy_scope = 'policy'
            with tf.variable_scope(policy_scope):
                ### create preprocess placeholders
                tf_preprocess = self._graph_preprocess_placeholders()
                ### process obs to lowd
                tf_obs_lowd = self._graph_obs_to_lowd(tf_obs_ph, tf_preprocess, is_training=True)
                ### create training policy
                tf_train_values, tf_train_values_softmax, _, _ = \
                    self._graph_inference(tf_obs_lowd, tf_actions_ph[:, :self._H, :],
                                          self._values_softmax, tf_preprocess, is_training=True)

            with tf.variable_scope(policy_scope, reuse=True):
                tf_train_values_test, tf_train_values_softmax_test, _, _ = \
                    self._graph_inference(tf_obs_lowd, tf_actions_ph[:, :self._get_action_test['H'], :],
                                          self._values_softmax, tf_preprocess, is_training=False)
                tf_get_value = tf.reduce_sum(tf_train_values_softmax_test * tf_train_values_test, reduction_indices=1)

            ### action selection
            tf_get_action, tf_get_action_value, tf_get_action_reset_ops = \
                self._graph_get_action(tf_obs_ph, self._get_action_test,
                                       policy_scope, True, policy_scope, True,
                                       tf_episode_timesteps_ph)
            ### exploration strategy and logprob
            tf_get_action_explore = self._graph_get_action_explore(tf_get_action, tf_test_es_ph_dict)

            ### get policy variables
            tf_policy_vars = sorted(tf.get_collection(xplatform.global_variables_collection_name(),
                                                      scope=policy_scope), key=lambda v: v.name)
            tf_trainable_policy_vars = sorted(tf.get_collection(xplatform.trainable_variables_collection_name(),
                                                                scope=policy_scope), key=lambda v: v.name)

            ##############
            ### target ###
            ##############

            target_scope = 'target'
            H_targ = N_targ = self._get_action_target['H']
            with tf.variable_scope(target_scope):
                ### create preprocess placeholders
                tf_preprocess = self._graph_preprocess_placeholders()
                ### process obs to lowd
                tf_obs_lowd = self._graph_obs_to_lowd(tf_obs_ph, tf_preprocess, is_training=True)
                ### create training policy
                tf_target_train_values, tf_target_train_values_softmax, _, _ = \
                    self._graph_inference(tf_obs_lowd, tf_actions_ph[:, :H_targ, :],
                                          self._values_softmax, tf_preprocess, is_training=True, N=N_targ)

            ### get target variables
            tf_target_vars = sorted(tf.get_collection(xplatform.global_variables_collection_name(),
                                                      scope=target_scope), key=lambda v: v.name)
            tf_trainable_target_vars = sorted(tf.get_collection(xplatform.trainable_variables_collection_name(),
                                                                scope=target_scope), key=lambda v: v.name)

            ### action selection
            target_target_scope = 'target_target'
            tf_obs_target_ph_packed = xplatform.concat([tf_obs_target_ph[:, h - self._obs_history_len:h, :]
                                                        for h in range(self._obs_history_len,
                                                                       self._obs_history_len + self._N + 1)],
                                                       0)
            tf_target_get_action, tf_target_get_action_values, _ = self._graph_get_action(tf_obs_target_ph_packed,
                                                                                          self._get_action_target,
                                                                                          scope_select=target_scope,
                                                                                          reuse_select=True,
                                                                                          scope_eval=target_target_scope,
                                                                                          reuse_eval=False,
                                                                                          tf_episode_timesteps_ph=None,  # TODO would need to fill in
                                                                                          N=N_targ)

            tf_target_get_action_values = tf.transpose(tf.reshape(tf_target_get_action_values, (self._N + 1, -1)))[:,1:]

            ### update target network
            tf_target_vars_nobatchnorm = list(filter(lambda v: 'biased' not in v.name and 'local_step' not in v.name,
                                                     tf_target_vars))
            tf_target_target_vars = sorted(tf.get_collection(xplatform.global_variables_collection_name(),
                                                      scope=target_target_scope), key=lambda v: v.name)
            assert (len(tf_target_vars_nobatchnorm) == len(tf_target_target_vars))
            tf_update_target_fn = []
            for var, var_target in zip(tf_target_vars_nobatchnorm, tf_target_target_vars):
                assert (var.name.replace(target_scope, '') == var_target.name.replace(target_target_scope, ''))
                tf_update_target_fn.append(var_target.assign(var))
            tf_update_target_fn = tf.group(*tf_update_target_fn)

            #########################
            ### Cost/optimization ###
            #########################

            ### policy
            tf_cost, tf_mse = self._graph_cost(tf_train_values, tf_train_values_softmax, tf_rewards_ph, tf_dones_ph,
                                               tf_target_get_action_values)
            tf_opt, tf_lr_ph = self._graph_optimize(tf_cost, tf_trainable_policy_vars)

            ### target
            tf_target_cost, tf_target_mse = self._graph_cost(tf_target_train_values, tf_target_train_values_softmax,
                                                             tf_rewards_ph[:, :N_targ], tf_dones_ph[:, :N_targ],
                                                             tf_target_get_action_values[:, :N_targ], N=N_targ)
            tf_target_opt, tf_target_lr_ph = self._graph_optimize(tf_target_cost, tf_trainable_target_vars)


            ### initialize
            self._graph_init_vars(tf_sess)

        ### what to return
        return {
            'sess': tf_sess,
            'graph': tf_graph,
            'obs_ph': tf_obs_ph,
            'actions_ph': tf_actions_ph,
            'dones_ph': tf_dones_ph,
            'rewards_ph': tf_rewards_ph,
            'obs_target_ph': tf_obs_target_ph,
            'test_es_ph_dict': tf_test_es_ph_dict,
            'episode_timesteps_ph': tf_episode_timesteps_ph,
            'preprocess': tf_preprocess,
            'get_value': tf_get_value,
            'get_action': tf_get_action,
            'get_action_explore': tf_get_action_explore,
            'get_action_value': tf_get_action_value,
            'get_action_reset_ops': tf_get_action_reset_ops,
            'update_target_fn': tf_update_target_fn,
            'cost': tf_cost,
            'cost_target': tf_target_cost,
            'mse': tf_mse,
            'mse_target': tf_target_mse,
            'opt': tf_opt,
            'opt_target': tf_target_opt,
            'lr_ph': tf_lr_ph,
            'lr_target_ph': tf_target_lr_ph,
            'policy_vars': tf_policy_vars,
            'target_vars': tf_target_vars
        }

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
        feed_dict = {
            ### parameters
            self._tf_dict['lr_ph']: self._lr_schedule.value(step),
            self._tf_dict['lr_target_ph']: self._lr_schedule.value(step),
            ### policy
            self._tf_dict['obs_ph']: observations[:, :self._obs_history_len, :],
            self._tf_dict['actions_ph']: actions,
            self._tf_dict['dones_ph']: np.logical_or(not use_target, dones[:, :self._N]),
            self._tf_dict['rewards_ph']: rewards[:, :self._N],
        }
        if self._use_target:
            feed_dict[self._tf_dict['obs_target_ph']] = observations

        cost, mse, cost_target, mse_target, _, _ = self._tf_dict['sess'].run([self._tf_dict['cost'],
                                                                              self._tf_dict['mse'],
                                                                              self._tf_dict['cost_target'],
                                                                              self._tf_dict['mse_target'],
                                                                              self._tf_dict['opt'],
                                                                              self._tf_dict['opt_target']],
                                                                             feed_dict=feed_dict)

        cost = 0.5 * (cost + cost_target)
        mse = 0.5 * (mse + mse_target)
        assert(np.isfinite(cost))

        self._log_stats['Cost'].append(cost)
        self._log_stats['mse/cost'].append(mse / cost)
