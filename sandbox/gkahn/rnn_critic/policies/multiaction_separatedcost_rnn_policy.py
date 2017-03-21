import tensorflow as tf
import tensorflow.contrib.layers as layers

from rllab.misc.overrides import overrides
from rllab.core.serializable import Serializable
from sandbox.gkahn.rnn_critic.policies.multiaction_combinedcost_rnn_policy import MultiactionCombinedcostRNNPolicy

class MultiactionSeparatedcostRNNPolicy(MultiactionCombinedcostRNNPolicy, Serializable):
    def __init__(self,
                 obs_hidden_layers,
                 action_hidden_layers,
                 reward_hidden_layers,
                 value_hidden_layers,
                 rnn_state_dim,
                 use_lstm,
                 activation,
                 rnn_activation,
                 **kwargs):
        """
        :param obs_hidden_layers: layer sizes for preprocessing the observation
        :param action_hidden_layers: layer sizes for preprocessing the action
        :param reward_hidden_layers: layer sizes for processing the reward
        :param value_hidden_layers: layer sizes for processing the value
        :param rnn_state_dim: dimension of the hidden state
        :param use_lstm: use lstm
        :param activation: string, e.g. 'tf.nn.relu'
        """
        Serializable.quick_init(self, locals())

        self._value_hidden_layers = list(value_hidden_layers)

        MultiactionCombinedcostRNNPolicy.__init__(self,
                                                  obs_hidden_layers=obs_hidden_layers,
                                                  action_hidden_layers=action_hidden_layers,
                                                  reward_hidden_layers=reward_hidden_layers,
                                                  rnn_state_dim=rnn_state_dim,
                                                  use_lstm=use_lstm,
                                                  activation=activation,
                                                  rnn_activation=rnn_activation,
                                                  **kwargs)

        assert(self._N > 1)
        assert(self._H > 1)
        assert(self._N == self._H)
        assert(self._cost_type == 'separated')

    ##################
    ### Properties ###
    ##################

    @property
    def N_output(self):
        return self._N + 1 # b/c of output value too

    ###########################
    ### TF graph operations ###
    ###########################

    @overrides
    def _graph_inference(self, tf_obs_ph, tf_actions_ph, d_preprocess):
        with tf.name_scope('inference'):
            tf_obs, tf_actions = self._graph_preprocess_inputs(tf_obs_ph, tf_actions_ph, d_preprocess)

            ### obs --> internal state
            with tf.name_scope('obs_to_istate'):
                layer = tf_obs
                for num_outputs in self._obs_hidden_layers + [self._rnn_state_dim]:
                    layer = layers.fully_connected(layer, num_outputs=num_outputs, activation_fn=self._activation,
                                                   weights_regularizer=layers.l2_regularizer(1.))
                istate = layer

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
                    rnn_cell = tf.nn.rnn_cell.BasicRNNCell(self._rnn_state_dim, activation=self._rnn_activation)
                    rnn_outputs, rnn_states = tf.nn.dynamic_rnn(rnn_cell, rnn_inputs, initial_state=istate)

            ### internal states --> reward sums
            with tf.name_scope('istates_to_reward_sums'):
                reward_sums = [0]
                for h in range(self._N):
                    layer = rnn_outputs[:, h, :]
                    for i, num_outputs in enumerate(self._reward_hidden_layers + [1]):
                        activation = self._activation if i < len(self._reward_hidden_layers) else None
                        layer = layers.fully_connected(layer,
                                                       num_outputs=num_outputs,
                                                       activation_fn=activation,
                                                       weights_regularizer=layers.l2_regularizer(1.),
                                                       scope='rewards_i{0}'.format(i),
                                                       reuse=(h > 0))
                    reward_sums.append(layer)

            ### last internal state --> terminal value sum
            with tf.name_scope('last_istate_to_value_sum'):
                layer = rnn_outputs[:, -1, :]
                for i, num_outputs in enumerate(self._value_hidden_layers + [1]):
                    activation = self._activation if i < len(self._value_hidden_layers) else None
                    layer = layers.fully_connected(layer,
                                                   num_outputs=num_outputs,
                                                   activation_fn=activation,
                                                   weights_regularizer=layers.l2_regularizer(1.),
                                                   scope='values_i{0}'.format(i),
                                                   reuse=False)
                value_sum = layer

            ### convert sums to rewards and value
            with tf.name_scope('sums_to_rewards_and_value'):
                rewards_value_sums = reward_sums + [value_sum]
                tf_rewards = []
                for i in range(len(rewards_value_sums) - 1):
                    tf_rewards.append(rewards_value_sums[i+1] - rewards_value_sums[i])
                tf_rewards = tf.concat(1, tf_rewards)



            tf_rewards = self._graph_preprocess_outputs(tf_rewards, d_preprocess)

        return tf_rewards

    @overrides
    def _graph_cost(self, tf_rewards_ph, tf_actions_ph, tf_rewards, tf_target_rewards, tf_target_mask_ph):
        # for training, len(tf_obs_ph) == len(tf_actions_ph)
        # but len(tf_actions_target_ph) == N * len(tf_action_ph),
        # so need to be selective about what to take the max over
        tf_target_values = self._graph_calculate_values(tf_target_rewards)
        batch_size = tf.shape(tf_rewards_ph)[0]
        tf_target_values_flat = tf.reshape(tf_target_values, (batch_size, -1))
        tf_target_values_max = tf.reduce_max(tf_target_values_flat, reduction_indices=1, keep_dims=True)

        if self._use_target:
            tf_rewards_ph_concat = tf.concat(1, [tf_rewards_ph, tf_target_values_max])
        else:
            tf_rewards_ph_concat = tf.concat(1, [tf_rewards_ph, tf.zeros((batch_size, 1))])
            tf_rewards = tf.concat(1, [tf_rewards[:, :-1], tf.zeros((batch_size, 1))])
        mse = tf.reduce_mean(self._graph_calculate_values(tf.square(tf_rewards - tf_rewards_ph_concat)))

        if len(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)) > 0:
            weight_decay = self._weight_decay * tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
        else:
            weight_decay = 0
        cost = mse + weight_decay
        return cost, mse
