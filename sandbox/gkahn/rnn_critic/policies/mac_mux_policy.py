import tensorflow as tf
import tensorflow.contrib.layers as layers
import numpy as np

from rllab.core.serializable import Serializable
from rllab.misc.overrides import overrides

from sandbox.rocky.tf.spaces.discrete import Discrete
from sandbox.gkahn.rnn_critic.policies.mac_policy import MACPolicy
from sandbox.gkahn.rnn_critic.tf.mux_rnn_cell import BasicMuxRNNCell, BasicMuxLSTMCell

class MACMuxPolicy(MACPolicy, Serializable):
    def __init__(self, **kwargs):
        Serializable.quick_init(self, locals())

        kwargs['action_hidden_layers'] = []
        kwargs['use_bilinear'] = False
        MACPolicy.__init__(self, **kwargs)

        assert(isinstance(self._env_spec.action_space, Discrete))

    ###########################
    ### TF graph operations ###
    ###########################

    @overrides
    def _graph_inference(self, tf_obs_lowd, tf_actions_ph, values_softmax, tf_preprocess, add_reg=True):
        """
        :param tf_obs_lowd: [batch_size, self._rnn_state_dim]
        :param tf_actions_ph: [batch_size, H, action_dim]
        :param values_softmax: string
        :param tf_preprocess:
        :return: tf_values: [batch_size, H]
        """
        H = tf_actions_ph.get_shape()[1].value
        tf.assert_equal(tf.shape(tf_obs_lowd)[0], tf.shape(tf_actions_ph)[0])
        action_dim = self._env_spec.action_space.flat_dim

        with tf.name_scope('inference'):
            ### internal state
            istate = tf_obs_lowd

            ### don't preprocess actions b/c discrete
            tf_actions = tf_actions_ph
            if H < self._N:
                tf_actions = tf.concat(1,
                                       [tf_actions,
                                        (1. / float(action_dim)) * tf.ones([tf.shape(tf_actions)[0], self._N - self._H, action_dim])])

            ### actions --> rnn input at each time step
            rnn_inputs = tf_actions

            ### create rnn
            with tf.name_scope('rnn'):
                with tf.variable_scope('rnn_vars'):
                    if self._use_lstm:
                        rnn_cell = BasicMuxLSTMCell(action_dim,
                                                    self._rnn_state_dim,
                                                    state_is_tuple=True,
                                                    activation=self._rnn_activation)
                        istate = tf.nn.rnn_cell.LSTMStateTuple(*tf.split(1, 2, istate))  # so state_is_tuple=True
                    else:
                        rnn_cell = BasicMuxRNNCell(action_dim, self._rnn_state_dim, activation=self._rnn_activation)
                    rnn_outputs, rnn_states = tf.nn.dynamic_rnn(rnn_cell, rnn_inputs, initial_state=istate)

            ### internal state --> nstep rewards
            with tf.name_scope('istates_to_nstep_rewards'):
                tf_nstep_rewards = [0]  # shifted by one
                for n in range(self._N - 1):  # shifted by one
                    layer = rnn_outputs[:, n, :]
                    for i, num_outputs in enumerate(self._reward_hidden_layers + [1]):
                        activation = self._activation if i < len(self._reward_hidden_layers) else None
                        layer = layers.fully_connected(layer,
                                                       num_outputs=num_outputs,
                                                       activation_fn=activation,
                                                       weights_regularizer=layers.l2_regularizer(1.) if add_reg else None,
                                                       scope='rewards_i{0}'.format(i),
                                                       reuse=tf.get_variable_scope().reuse or (n > 0))
                    tf_nstep_rewards.append(layer)

            ### internal state --> nstep values
            with tf.name_scope('istates_to_nstep_values'):
                tf_nstep_values = []
                for n in range(self._N):
                    layer = rnn_outputs[:, n, :]
                    for i, num_outputs in enumerate(self._value_hidden_layers + [1]):
                        activation = self._activation if i < len(self._value_hidden_layers) else None
                        layer = layers.fully_connected(layer,
                                                       num_outputs=num_outputs,
                                                       activation_fn=activation,
                                                       weights_regularizer=layers.l2_regularizer(1.) if add_reg else None,
                                                       scope='values_i{0}'.format(i),
                                                       reuse=tf.get_variable_scope().reuse or (n > 0))
                    tf_nstep_values.append(layer)

            ### internal state --> nstep lambdas
            with tf.name_scope('istates_to_nstep_lambdas'):
                tf_nstep_lambdas = []
                for n in range(self._N - 1):  # since must use last value
                    layer = rnn_outputs[:, n, :]
                    for i, num_outputs in enumerate(self._lambda_hidden_layers + [1]):
                        activation = self._activation if i < len(self._lambda_hidden_layers) else tf.sigmoid
                        layer = layers.fully_connected(layer,
                                                       num_outputs=num_outputs,
                                                       activation_fn=activation,
                                                       weights_regularizer=layers.l2_regularizer(1.) if add_reg else None,
                                                       scope='lambdas_i{0}'.format(i),
                                                       reuse=tf.get_variable_scope().reuse or (n > 0))
                    tf_nstep_lambdas.append(layer)
                tf_nstep_lambdas.append(tf.zeros([tf.shape(layer)[0], 1]))  # since must use last value

            ### nstep rewards + nstep values --> values
            with tf.name_scope('nstep_rewards_nstep_values_to_values'):
                tf_values_list = []
                for n in range(self._N):
                    tf_returns = tf_nstep_rewards[:n] + [tf_nstep_values[n]]
                    tf_values_list.append(
                        np.sum([np.power(self._gamma, i) * tf_return for i, tf_return in enumerate(tf_returns)]))
                tf_values = tf.concat(1, tf_values_list)

            ### nstep lambdas --> values softmax and depth
            with tf.name_scope('nstep_lambdas_to_values_softmax_and_depth'):
                if values_softmax == 'final':
                    tf_values_softmax = tf.one_hot(self._N - 1, self._N) * tf.ones(tf.shape(tf_values))
                    tf_values_depth = (self._N - 1) * tf.ones([tf.shape(tf_values)[0]])
                elif values_softmax == 'mean':
                    tf_values_softmax = (1. / float(self._N)) * tf.ones(tf.shape(tf_values))
                    tf_values_depth = ((self._N - 1) / 2.) * tf.ones([tf.shape(tf_values)[0]])
                elif values_softmax == 'learned':
                    tf_values_softmax_list = []
                    for n in range(self._N):
                        tf_values_softmax_list.append(np.prod(tf_nstep_lambdas[:n]) * (1 - tf_nstep_lambdas[n]))
                    tf_values_softmax = tf.concat(1, tf_values_softmax_list)
                    tf.assert_less(tf.reduce_max(tf.reduce_sum(tf_values_softmax, 1)), 1.0001)
                    tf.assert_greater(tf.reduce_min(tf.reduce_sum(tf_values_softmax, 1)), 0.999)

                    curr_depth = 0
                    for n in range(self._N - 1, -1, -1):
                        curr_depth = tf_nstep_lambdas[n] * (1 + np.power(self._gamma, n) * curr_depth)
                    tf_values_depth = curr_depth

        assert (tf_values.get_shape()[1].value == self._N)

        return tf_values, tf_values_softmax, tf_values_depth
