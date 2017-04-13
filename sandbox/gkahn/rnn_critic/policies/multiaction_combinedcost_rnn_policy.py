import tensorflow as tf
import tensorflow.contrib.layers as layers

from rllab.misc.overrides import overrides
from rllab.core.serializable import Serializable
from sandbox.gkahn.rnn_critic.policies.policy import Policy
from sandbox.gkahn.rnn_critic.tf.mulint_rnn_cell import BasicMulintRNNCell, BasicMulintLSTMCell

class MultiactionCombinedcostRNNPolicy(Policy, Serializable):
    def __init__(self,
                 obs_hidden_layers,
                 action_hidden_layers,
                 reward_hidden_layers,
                 rnn_state_dim,
                 use_lstm,
                 use_bilinear,
                 activation,
                 rnn_activation,
                 conv_hidden_layers=None,
                 conv_kernels=None,
                 conv_strides=None,
                 conv_activation=None,
                 **kwargs):
        """
        :param obs_hidden_layers: layer sizes for preprocessing the observation
        :param action_hidden_layers: layer sizes for preprocessing the action
        :param reward_hidden_layers: layer sizes for processing the reward
        :param rnn_state_dim: dimension of the hidden state
        :param use_lstm: use lstm
        :param use_bilinear: use multiplicative integration?
        :param activation: string, e.g. 'tf.nn.relu'
        """
        Serializable.quick_init(self, locals())

        self._obs_hidden_layers = list(obs_hidden_layers)
        self._action_hidden_layers = list(action_hidden_layers)
        self._reward_hidden_layers = list(reward_hidden_layers)
        self._rnn_state_dim = rnn_state_dim
        self._use_lstm = use_lstm
        self._use_bilinear = use_bilinear
        self._activation = eval(activation)
        self._rnn_activation = eval(rnn_activation)
        self._use_conv = (conv_hidden_layers is not None) and (conv_kernels is not None) and \
                         (conv_strides is not None) and (conv_activation is not None)
        if self._use_conv:
            self._conv_hidden_layers = list(conv_hidden_layers)
            self._conv_kernels = list(conv_kernels)
            self._conv_strides = list(conv_strides)
            self._conv_activation = eval(conv_activation)

        Policy.__init__(self, **kwargs)

        # assert(self._N > 1)
        # assert(self._H > 1)
        assert(self._N == self._H)

    ##################
    ### Properties ###
    ##################

    @property
    def N_output(self):
        return 1

    ###########################
    ### TF graph operations ###
    ###########################

    @overrides
    def _graph_inference(self, tf_obs_ph, tf_actions_ph, d_preprocess):
        with tf.name_scope('inference'):
            tf_obs, tf_actions = self._graph_preprocess_inputs(tf_obs_ph, tf_actions_ph, d_preprocess)

            ### obs --> lower dimensional space
            with tf.name_scope('obs_to_lowd'):
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

            ### obs --> internal state
            with tf.name_scope('obs_to_istate'):
                final_dim = self._rnn_state_dim if not self._use_lstm else 2 * self._rnn_state_dim
                for num_outputs in self._obs_hidden_layers + [final_dim]:
                    layer = layers.fully_connected(layer, num_outputs=num_outputs, activation_fn=self._activation,
                                                   weights_regularizer=layers.l2_regularizer(1.))
                istate = layer

            ### replicate istate if needed
            istate = self._graph_match_actions(istate, tf_actions)

            ### actions --> rnn input at each time step
            with tf.name_scope('actions_to_rnn_input'):
                tf_actions_list = tf.split(1, self._N, tf_actions)
                rnn_inputs = []
                for h in range(self._N):
                    layer = tf_actions_list[h]

                    for i, num_outputs in enumerate(self._action_hidden_layers + [self._rnn_state_dim]):
                        layer = layers.fully_connected(layer, num_outputs=num_outputs, activation_fn=self._activation,
                                                       weights_regularizer=layers.l2_regularizer(1.),
                                                       scope='actions_i{0}'.format(i),
                                                       reuse=(h > 0))
                    rnn_inputs.append(layer)
                rnn_inputs = tf.pack(rnn_inputs, 1)

            ### create rnn
            with tf.name_scope('rnn'):
                with tf.variable_scope('rnn_vars'):
                    if self._use_lstm:
                        if self._use_bilinear:
                            rnn_cell = BasicMulintLSTMCell(self._rnn_state_dim,
                                                           state_is_tuple=True,
                                                           activation=self._rnn_activation)
                        else:
                            rnn_cell = tf.nn.rnn_cell.BasicLSTMCell(self._rnn_state_dim,
                                                                    state_is_tuple=True,
                                                                    activation=self._rnn_activation)
                        istate = tf.nn.rnn_cell.LSTMStateTuple(*tf.split(1, 2, istate))  # so state_is_tuple=True
                    else:
                        if self._use_bilinear:
                            rnn_cell = BasicMulintRNNCell(self._rnn_state_dim, activation=self._rnn_activation)
                        else:
                            rnn_cell = tf.nn.rnn_cell.BasicRNNCell(self._rnn_state_dim, activation=self._rnn_activation)
                    rnn_outputs, rnn_states = tf.nn.dynamic_rnn(rnn_cell, rnn_inputs, initial_state=istate)

            ### final internal state --> reward
            with tf.name_scope('final_istate_to_reward'):
                layer = rnn_outputs[:, -1, :]
                for i, num_outputs in enumerate(self._reward_hidden_layers + [1]):
                    activation = self._activation if i < len(self._reward_hidden_layers) else None
                    layer = layers.fully_connected(layer,
                                                   num_outputs=num_outputs,
                                                   activation_fn=activation,
                                                   weights_regularizer=layers.l2_regularizer(1.),
                                                   scope='rewards_i{0}'.format(i),
                                                   reuse=False)
                tf_rewards = layer

            tf_rewards = self._graph_preprocess_outputs(tf_rewards, d_preprocess)

        return tf_rewards
