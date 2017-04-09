import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers

from rllab.misc.overrides import overrides
from rllab.core.serializable import Serializable
from sandbox.gkahn.rnn_critic.policies.policy import Policy
from sandbox.gkahn.rnn_critic.utils import tf_utils

class DiscreteDQNPolicy(Policy, Serializable):
    def __init__(self,
                 hidden_layers,
                 activation,
                 conv_hidden_layers=None,
                 conv_kernels=None,
                 conv_strides=None,
                 conv_activation=None,
                 **kwargs):
        """
        :param hidden_layers: list of layer sizes
        :param activation: str to be evaluated (e.g. 'tf.nn.relu')
        """
        Serializable.quick_init(self, locals())

        self._hidden_layers = list(hidden_layers)
        self._activation = eval(activation)
        self._use_conv = (conv_hidden_layers is not None) and (conv_kernels is not None) and \
                         (conv_strides is not None) and (conv_activation is not None)
        if self._use_conv:
            self._conv_hidden_layers = list(conv_hidden_layers)
            self._conv_kernels = list(conv_kernels)
            self._conv_strides = list(conv_strides)
            self._conv_activation = eval(conv_activation)

        Policy.__init__(self, **kwargs)

        assert(self._H == 1)
        assert(self._cost_type == 'combined')

    ##################
    ### Properties ###
    ##################

    @property
    def N_output(self):
        return self._env_spec.action_space.flat_dim

    ###########################
    ### TF graph operations ###
    ###########################

    @overrides
    def _graph_preprocess_inputs(self, tf_obs_ph, d_preprocess):
        ### whiten inputs
        if tf_obs_ph.dtype != tf.float32:
            tf_obs_ph = tf.cast(tf_obs_ph, tf.float32)
        if self._obs_is_im:
            tf_obs_whitened = tf.mul(tf_obs_ph -
                                     tf.tile(d_preprocess['observations_mean_var'], (1, self._obs_history_len)),
                                     tf.tile(d_preprocess['observations_orth_var'], (self._obs_history_len,)))
        else:
            tf_obs_whitened = tf.matmul(tf_obs_ph -
                                        tf.tile(d_preprocess['observations_mean_var'], (1, self._obs_history_len)),
                                        tf_utils.block_diagonal([d_preprocess['observations_orth_var']] * self._obs_history_len))

        return tf_obs_whitened

    @overrides
    def _graph_preprocess_outputs(self, tf_rewards, d_preprocess):
        # return (tf_rewards * d_preprocess['rewards_orth_var'][0, 0]) + d_preprocess['rewards_mean_var']
        return tf_rewards # don't preprocess outputs

    @overrides
    def _graph_inference(self, tf_obs_ph, tf_actions_ph, d_preprocess):
        output_dim = self.N_output

        with tf.name_scope('inference'):
            tf_obs = self._graph_preprocess_inputs(tf_obs_ph, d_preprocess)

            if self._use_conv:
                obs_shape = [self._obs_history_len] + list(self._env_spec.observation_space.shape)[:2]
                layer = tf.transpose(tf.reshape(tf_obs, [-1] + list(obs_shape)), perm=(0, 2, 3, 1))
                for num_outputs, kernel_size, stride in zip(self._conv_hidden_layers,
                                                            self._conv_kernels,
                                                            self._conv_strides):
                    layer = layers.convolution2d(layer,
                                                 num_outputs=num_outputs,
                                                 kernel_size=kernel_size,
                                                 stride=stride,
                                                 activation_fn=self._conv_activation)
                layer = layers.flatten(layer)
            else:
                layer = layers.flatten(tf_obs)

            ### fully connected
            for num_outputs in self._hidden_layers:
                layer = layers.fully_connected(layer, num_outputs=num_outputs, activation_fn=self._activation,
                                               weights_regularizer=layers.l2_regularizer(1.))
            layer = layers.fully_connected(layer, num_outputs=output_dim, activation_fn=None,
                                           weights_regularizer=layers.l2_regularizer(1.))

            tf_rewards = self._graph_preprocess_outputs(layer, d_preprocess)

            # num_vars = np.sum([np.prod(v.get_shape()) for v in tf.trainable_variables()])
            # print('num_vars: {0}'.format(num_vars))
            # import IPython; IPython.embed()

        return tf_rewards

    @overrides
    def _graph_calculate_values(self, tf_rewards):
        return tf_rewards # b/c network outputs values directly

    @overrides
    def _graph_cost(self, tf_rewards_ph, tf_actions_ph, tf_rewards,
                    tf_target_rewards_select, tf_target_rewards_eval, tf_target_mask_ph):
        ### target: calculate values
        tf_target_values_select = self._graph_calculate_values(tf_target_rewards_select)
        tf_target_values_eval = self._graph_calculate_values(tf_target_rewards_eval)
        ### target: mask selection and eval
        tf_target_values_mask = tf.one_hot(tf.argmax(tf_target_values_select, 1),
                                           depth=self.N_output)
        tf_target_values_max = tf.reduce_sum(tf_target_values_mask * tf_target_values_eval, reduction_indices=1)

        ### policy:
        tf_values_ph = tf.reduce_sum(np.power(self._gamma, np.arange(self._N)) * tf_rewards_ph, reduction_indices=1)
        tf_values = tf.reduce_sum(tf_actions_ph * tf_rewards, reduction_indices=1)

        mse = tf.reduce_mean(tf.square(tf_values_ph +
                                       self._use_target * tf_target_mask_ph * np.power(self._gamma, self._N) * tf_target_values_max -
                                       tf_values))
        if len(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)) > 0:
            weight_decay = self._weight_decay * tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
        else:
            weight_decay = 0
        cost = mse + weight_decay

        return cost, mse

    ################
    ### Training ###
    ################

    @overrides
    def update_preprocess(self, preprocess_stats):
        for key in ('actions_mean', 'actions_orth',
                    'rewards_mean', 'rewards_orth'):
            assert(self._preprocess_params[key] is False)
        Policy.update_preprocess(self, preprocess_stats)

    @overrides
    def train_step(self, step, observations, actions, rewards, dones, use_target):
        batch_size = len(observations)
        action_dim = self._env_spec.action_space.flat_dim

        policy_observations = observations[:, 0, :]
        policy_actions = actions[:, :self._H, :].reshape((batch_size, self._H * action_dim))
        policy_rewards = rewards[:, :self._N]
        target_observations = observations[:, self._N, :]

        feed_dict = {
            ### parameters
            self._tf_lr_ph: self._lr_schedule.value(step),
            ### policy
            self._tf_obs_ph: policy_observations,
            self._tf_actions_ph: policy_actions,
            self._tf_rewards_ph: policy_rewards,
            ### target network
            self._tf_obs_target_ph: target_observations,
            self._tf_target_mask_ph: float(use_target) * (1 - dones[:, self._N].astype(float))
        }
        cost, mse, _ = self._tf_sess.run([self._tf_cost, self._tf_mse, self._tf_opt], feed_dict=feed_dict)

        assert (np.isfinite(cost))

        self._log_stats['Cost'].append(cost)
        self._log_stats['mse/cost'].append(mse / cost)

    ######################
    ### Policy methods ###
    ######################

    @overrides
    def get_actions(self, observations, return_action_info=False):
        action_dim = self._env_spec.action_space.flat_dim
        num_obs = len(observations)
        observations = self._env_spec.observation_space.flatten_n(observations)

        pred_values = self._tf_sess.run(self._tf_values,
                                        feed_dict={self._tf_obs_ph: observations})

        chosen_actions = []
        for i, (observation_i, pred_values_i) in enumerate(zip(observations, pred_values)):

            chosen_action_i = int(pred_values_i.argmax())

            if self._exploration_strategy is not None:
                exploration_func = lambda: None
                exploration_func.get_action = lambda _: (chosen_action_i, dict())
                chosen_action_i = self._exploration_strategy.get_action(self._num_get_action + i,
                                                                        observation_i,
                                                                        exploration_func)
            chosen_actions.append(chosen_action_i)

        self._num_get_action += num_obs

        if return_action_info:
            action_info = {
                'values': pred_values
            }
        else:
            action_info = dict()

        return chosen_actions, action_info
