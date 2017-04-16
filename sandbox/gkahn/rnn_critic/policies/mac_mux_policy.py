import tensorflow as tf
import tensorflow.contrib.layers as layers

from rllab.core.serializable import Serializable

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

    def _graph_inference(self, tf_obs_lowd, tf_actions_ph, tf_preprocess):
        """
        :param tf_obs_lowd: [batch_size, self._rnn_state_dim]
        :param tf_actions_ph: [batch_size, H, action_dim]
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

            ### final internal state --> reward
            with tf.name_scope('final_istate_to_reward'):
                tf_values_list = []
                for h in range(H):
                    layer = rnn_outputs[:, h, :]
                    for i, num_outputs in enumerate(self._reward_hidden_layers + [1]):
                        activation = self._activation if i < len(self._reward_hidden_layers) else None
                        layer = layers.fully_connected(layer,
                                                       num_outputs=num_outputs,
                                                       activation_fn=activation,
                                                       weights_regularizer=layers.l2_regularizer(1.),
                                                       scope='rewards_i{0}'.format(i),
                                                       reuse=tf.get_variable_scope().reuse or (h > 0))
                    ### de-whiten
                    tf_values_h = tf.add(tf.matmul(layer, tf.transpose(tf_preprocess['rewards_orth_var'])),
                                         tf_preprocess['rewards_mean_var'])
                    tf_values_list.append(tf_values_h)

                tf_values = tf.concat(1, tf_values_list)

        assert(tf_values.get_shape()[1].value == H)

        return tf_values
