import numpy as np
import tensorflow as tf

from rllab.core.serializable import Serializable
from rllab.misc.overrides import overrides
from rllab.misc import ext

from sandbox.gkahn.rnn_critic.policies.mac_policy import MACPolicy
from sandbox.gkahn.tf.core import xplatform

class NotargetMACPolicy(MACPolicy, Serializable):
    def __init__(self, **kwargs):
        Serializable.quick_init(self, locals())

        self._consistency_weight = kwargs['consistency_weight']

        MACPolicy.__init__(self, **kwargs)

        assert(self.only_completed_episodes)

    ###########################
    ### TF graph operations ###
    ###########################

    @overrides
    def _graph_input_output_placeholders(self):
        obs_shape = self._env_spec.observation_space.shape
        obs_dtype = tf.uint8 if len(obs_shape) > 1 else tf.float32
        obs_dim = self._env_spec.observation_space.flat_dim
        action_dim = self._env_spec.action_space.flat_dim

        with tf.variable_scope('input_output_placeholders'):
            ### policy inputs
            tf_obs_ph = tf.placeholder(obs_dtype, [None, self._obs_history_len, obs_dim], name='tf_obs_ph')
            tf_actions_ph = tf.placeholder(tf.float32, [None, self._N + 1, action_dim], name='tf_actions_ph')
            tf_dones_ph = tf.placeholder(tf.bool, [None, self._N], name='tf_dones_ph')
            ### policy outputs
            tf_rewards_ph = tf.placeholder(tf.float32, [None, self._N], name='tf_rewards_ph')
            tf_values_ph = tf.placeholder(tf.float32, [None, self._N], name='tf_values_ph')
            ### policy exploration
            tf_explore_train_ph = tf.placeholder(tf.float32, [None, self._N + 1], name='tf_explore_train')
            tf_explore_test_ph = tf.placeholder(tf.float32, [None], name='tf_explore_test')
            ### importance sampling
            tf_logprob_prior_ph = tf.placeholder(tf.float32, [None, self._N], name='tf_logprob_prior_ph')

        return tf_obs_ph, tf_actions_ph, tf_dones_ph, tf_rewards_ph, tf_values_ph, \
               tf_explore_train_ph, tf_explore_test_ph, tf_logprob_prior_ph

    @overrides
    def _graph_cost(self, tf_train_nstep_rewards, tf_train_nstep_values, tf_train_values_softmax, tf_rewards_ph, tf_values_ph, tf_dones_ph):
        """
        :param tf_train_nstep_rewards: list of [None] of length self._N
        :param tf_train_nstep_values: list of [None] of length self._N
        :param tf_train_values_softmax: [None, self._N]
        :param tf_rewards_ph: [None, self._N]
        :param tf_values_ph: [None, self._N]
        :param tf_dones_ph: [None, self._N]
        :return: tf_cost, tf_mse
        """
        tf_rewards_ph = tf.split(1, self._N, tf_rewards_ph)
        tf_values_ph = tf.split(1, self._N, tf_values_ph)
        tf_dones_ph = tf.cast(tf_dones_ph, tf.float32)

        assert(self._retrace_lambda is None)
        tf.assert_equal(tf.reduce_sum(tf_train_values_softmax, 1), 1.)

        tf_values = []
        values = []
        for n in range(self._N):
            tf_sum_rewards_n = np.sum([np.power(self._gamma, i) * r for i, r in enumerate(tf_train_nstep_rewards[:n])])
            sum_rewards_n = np.sum([np.power(self._gamma, i) * r for i, r in enumerate(tf_rewards_ph[:n])])

            tf_values_n = np.power(self._gamma, n) * tf_train_nstep_values[n]
            values_n = np.power(self._gamma, n) * tf_values_ph[n]

            dones_n = tf.expand_dims(tf_dones_ph[:, n], 1)

            if n == 0:
                tf_values.append(tf_values_n)
                values.append((1 - dones_n) * values_n)
            else:
                tf_values.append(tf_sum_rewards_n + tf_values_n)
                values.append(sum_rewards_n + (1 - dones_n) * values_n)

        assert(self._separate_mses)
        tf_mse = tf.reduce_mean(tf.reduce_sum(tf_train_values_softmax * tf.square(tf.concat(1, tf_values) - tf.concat(1, values)), 1))
        tf_consistency = self._consistency_weight * \
                         tf.reduce_mean((1. / float(self._N)) * np.sum(
                             [tf.square(tf_values[i+1] - tf_values[i]) for i in range(self._N - 1)]))

        ### weight decay
        if len(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)) > 0:
            tf_weight_decay = self._weight_decay * tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
        else:
            tf_weight_decay = 0
        tf_cost = tf_mse + tf_consistency + tf_weight_decay

        return tf_cost, tf_mse

    @overrides
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
            tf_obs_ph, tf_actions_ph, tf_dones_ph, tf_rewards_ph, tf_values_ph, \
            tf_explore_train_ph, tf_explore_test_ph, tf_logprob_prior_ph = self._graph_input_output_placeholders()

            ### policy
            policy_scope = 'policy'
            with tf.variable_scope(policy_scope):
                ### create preprocess placeholders
                tf_preprocess = self._graph_preprocess_placeholders()
                ### process obs to lowd
                tf_obs_lowd = self._graph_obs_to_lowd(tf_obs_ph, tf_preprocess)
                ### create training policy
                _, tf_train_values_softmax, tf_train_nstep_rewards, tf_train_nstep_values = \
                    self._graph_inference(tf_obs_lowd, tf_actions_ph[:, :self._H, :],
                                          self._values_softmax, tf_preprocess)

            with tf.variable_scope(policy_scope, reuse=True):
                tf_train_values_test, tf_train_values_softmax_test, _, _ = \
                    self._graph_inference(tf_obs_lowd, tf_actions_ph[:, :self._get_action_test['H'], :],
                                          self._values_softmax, tf_preprocess)
                tf_get_value = tf.reduce_sum(tf_train_values_softmax_test * tf_train_values_test, reduction_indices=1)

            ### action selection
            tf_get_action, tf_get_action_value = self._graph_get_action(tf_obs_ph, self._get_action_test,
                                                                        policy_scope, True, policy_scope, True)
            ### exploration strategy and logprob
            tf_get_action_explore = self._graph_get_action_explore(tf_get_action, tf_explore_test_ph)
            tf_get_action_logprob = self._graph_get_action_logprob(tf_get_action_explore, tf_get_action, tf_explore_test_ph)

            ### get policy variables
            tf_trainable_policy_vars = sorted(tf.get_collection(xplatform.trainable_variables_collection_name(),
                                                                scope=policy_scope), key=lambda v: v.name)

            assert (self._retrace_lambda is None)

            ### update target network
            tf_update_target_fn = None

            ### optimization
            tf_cost, tf_mse = self._graph_cost(tf_train_nstep_rewards, tf_train_nstep_values,
                                               tf_train_values_softmax, tf_rewards_ph, tf_values_ph, tf_dones_ph)
            tf_opt, tf_lr_ph = self._graph_optimize(tf_cost, tf_trainable_policy_vars)

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
            'values_ph': tf_values_ph,
            'explore_train_ph': tf_explore_train_ph,
            'explore_test_ph': tf_explore_test_ph,
            'logprob_ph': tf_logprob_prior_ph,
            'preprocess': tf_preprocess,
            'get_value': tf_get_value,
            'get_action': tf_get_action,
            'get_action_explore': tf_get_action_explore,
            'get_action_value': tf_get_action_value,
            'get_action_logprob': tf_get_action_logprob,
            'update_target_fn': tf_update_target_fn,
            'cost': tf_cost,
            'mse': tf_mse,
            'opt': tf_opt,
            'lr_ph': tf_lr_ph,
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
            self._tf_dict['explore_train_ph']: np.reshape([self._exploration_strategy.schedule.value(s)
                                                           for s in steps.ravel()], steps.shape),
            ### policy
            self._tf_dict['obs_ph']: observations[:, :self._obs_history_len, :],
            self._tf_dict['actions_ph']: actions,
            self._tf_dict['dones_ph']: np.logical_or(not use_target, dones[:, :self._N]),
            self._tf_dict['rewards_ph']: rewards[:, :self._N],
            self._tf_dict['values_ph']: values[:, :self._N]
        }

        cost, mse, _ = self._tf_dict['sess'].run([self._tf_dict['cost'],
                                                  self._tf_dict['mse'],
                                                  self._tf_dict['opt']],
                                                 feed_dict=feed_dict)
        assert(np.isfinite(cost))

        self._log_stats['Cost'].append(cost)
        self._log_stats['mse/cost'].append(mse / cost)
