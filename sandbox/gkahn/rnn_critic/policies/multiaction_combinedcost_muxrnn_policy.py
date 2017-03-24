import tensorflow as tf
import tensorflow.contrib.layers as layers

from rllab.misc.overrides import overrides
from rllab.core.serializable import Serializable

from sandbox.gkahn.rnn_critic.policies.policy import Policy
from sandbox.gkahn.rnn_critic.tf.mux_rnn_cell import BasicMuxRNNCell, BasicMuxLSTMCell

class MultiactionCombinedcostMuxRNNPolicy(Policy, Serializable):
    def __init__(self,
                 obs_hidden_layers,
                 reward_hidden_layers,
                 rnn_state_dim,
                 use_lstm,
                 activation,
                 rnn_activation,
                 **kwargs):
        """
        :param obs_hidden_layers: layer sizes for preprocessing the observation
        :param action_hidden_layers: layer sizes for preprocessing the action
        :param reward_hidden_layers: layer sizes for processing the reward
        :param rnn_state_dim: dimension of the hidden state
        :param use_lstm: use lstm
        :param activation: string, e.g. 'tf.nn.relu'
        """
        Serializable.quick_init(self, locals())

        self._obs_hidden_layers = list(obs_hidden_layers)
        self._reward_hidden_layers = list(reward_hidden_layers)
        self._rnn_state_dim = rnn_state_dim
        self._use_lstm = use_lstm
        self._activation = eval(activation)
        self._rnn_activation = eval(rnn_activation)

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
        ac_dim = self._env_spec.action_space.flat_dim

        with tf.name_scope('inference'):
            tf_obs, tf_actions = self._graph_preprocess_inputs(tf_obs_ph, tf_actions_ph, d_preprocess)

            ### obs --> internal state
            with tf.name_scope('obs_to_istate'):
                layer = tf_obs
                final_dim = self._rnn_state_dim if not self._use_lstm else 2 * self._rnn_state_dim
                for num_outputs in self._obs_hidden_layers + [final_dim]:
                    layer = layers.fully_connected(layer, num_outputs=num_outputs, activation_fn=self._activation,
                                                   weights_regularizer=layers.l2_regularizer(1.))
                istate = layer

            ### actions --> rnn input at each time step
            with tf.name_scope('actions_to_rnn_input'):
                rnn_inputs = tf.reshape(tf_actions, (-1, self._N, ac_dim))

            ### create rnn
            with tf.name_scope('rnn'):
                with tf.variable_scope('rnn_vars'):
                    if self._use_lstm:
                        rnn_cell = BasicMuxLSTMCell(ac_dim,
                                                    self._rnn_state_dim,
                                                    state_is_tuple=True,
                                                    activation=self._rnn_activation)
                        istate = tf.nn.rnn_cell.LSTMStateTuple(*tf.split(1, 2, istate)) # so state_is_tuple=True
                    else:
                        rnn_cell = BasicMuxRNNCell(ac_dim, self._rnn_state_dim, activation=self._rnn_activation)
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
