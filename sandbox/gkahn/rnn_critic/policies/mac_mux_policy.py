import tensorflow as tf
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
    def _graph_inference_step(self, n, N, istate, action, values_softmax, add_reg=True):
        """
        :param n: current step
        :param N: max step
        :param istate: current internal state
        :param action: if action is None, input zeros
        """
        import tensorflow.contrib.layers as layers
        batch_size = tf.shape(istate)[0]
        action_dim = self._env_spec.action_space.flat_dim

        with tf.name_scope('inference_step'):
            ### action
            with tf.name_scope('action'):
                if action is None:
                    rnn_input = (1. / float(action_dim)) * tf.ones([batch_size, action_dim])
                else:
                    rnn_input = action

            ### rnn
            with tf.name_scope('rnn'):
                scope = 'rnn' if self._share_weights else 'rnn_n{0}'.format(n)
                with tf.variable_scope(scope):
                    if self._use_lstm:
                        rnn_cell = BasicMuxLSTMCell(action_dim,
                                                    self._rnn_state_dim,
                                                    state_is_tuple=True,
                                                    activation=self._rnn_activation)
                    else:
                        rnn_cell = BasicMuxRNNCell(action_dim, self._rnn_state_dim, activation=self._rnn_activation)
                    tf_output, next_istate = rnn_cell(rnn_input, istate)

            ### rnn output --> nstep rewards
            with tf.name_scope('nstep_rewards'):
                layer = tf_output
                for i, num_outputs in enumerate(self._reward_hidden_layers + [1]):
                    activation = self._activation if i < len(self._reward_hidden_layers) else None
                    scope = 'rewards_i{0}'.format(i) if self._share_weights else 'rewards_n{0}_i{1}'.format(n, i)
                    layer = layers.fully_connected(layer,
                                                   num_outputs=num_outputs,
                                                   activation_fn=activation,
                                                   weights_regularizer=layers.l2_regularizer(1.) if add_reg else None,
                                                   scope=scope,
                                                   reuse=tf.get_variable_scope().reuse)
                tf_nstep_reward = layer

            ### rnn output --> nstep values
            with tf.name_scope('nstep_values'):
                layer = tf_output
                for i, num_outputs in enumerate(self._value_hidden_layers + [1]):
                    activation = self._activation if i < len(self._value_hidden_layers) else None
                    scope = 'values_i{0}'.format(i) if self._share_weights else 'values_n{0}_i{1}'.format(n, i)
                    layer = layers.fully_connected(layer,
                                                   num_outputs=num_outputs,
                                                   activation_fn=activation,
                                                   weights_regularizer=layers.l2_regularizer(1.) if add_reg else None,
                                                   scope=scope,
                                                   reuse=tf.get_variable_scope().reuse)
                tf_nstep_value = layer

            ### nstep lambdas --> values softmax and depth
            batch_size = tf.shape(tf_nstep_value)[0]
            with tf.name_scope('nstep_lambdas'):
                if values_softmax['type'] == 'final':
                    if n == N - 1:
                        tf_values_softmax = tf.ones([batch_size])
                    else:
                        tf_values_softmax = tf.zeros([batch_size])
                elif values_softmax['type'] == 'mean':
                    tf_values_softmax = (1. / float(N)) * tf.ones([batch_size])
                elif values_softmax['type'] == 'exponential':
                    lam = values_softmax['exponential']['lambda']
                    if n == N - 1:
                        tf_values_softmax = np.power(lam, n) * tf.ones([batch_size])
                    else:
                        tf_values_softmax = (1 - lam) * np.power(lam, n) * tf.ones([batch_size])
                else:
                    raise NotImplementedError

        return next_istate, tf_nstep_reward, tf_nstep_value, tf_values_softmax

