import tensorflow as tf

from rllab.misc.overrides import overrides
from rllab.core.serializable import Serializable
from sandbox.gkahn.rnn_critic.policies.policy import RNNCriticPolicy

class RNNCriticRNNPolicy(RNNCriticPolicy, Serializable):
    def __init__(self,
                 obs_hidden_layers,
                 action_hidden_layers,
                 reward_hidden_layers,
                 rnn_state_dim,
                 activation,
                 rnn_activation,
                 **kwargs):
        """
        :param obs_hidden_layers: layer sizes for preprocessing the observation
        :param action_hidden_layers: layer sizes for preprocessing the action
        :param reward_hidden_layers: layer sizes for processing the reward
        :param rnn_state_dim: dimension of the hidden state
        :param activation: string, e.g. 'tf.nn.relu'
        """
        Serializable.quick_init(self, locals())

        self._obs_hidden_layers = list(obs_hidden_layers)
        self._action_hidden_layers = list(action_hidden_layers)
        self._reward_hidden_layers = list(reward_hidden_layers)
        self._rnn_state_dim = rnn_state_dim
        self._activation = eval(activation)
        self._rnn_activation = eval(rnn_activation)

        RNNCriticPolicy.__init__(self, **kwargs)

    @overrides
    def _graph_inference(self, tf_obs_ph, tf_actions_ph, d_preprocess):
        is_lstm = False # TODO

        obs_dim = self._env_spec.observation_space.flat_dim
        action_dim = self._env_spec.action_space.flat_dim

        with tf.name_scope('inference'):
            tf_obs, tf_actions = self._graph_preprocess_inputs(tf_obs_ph, tf_actions_ph, d_preprocess)

            ########################
            ### Create variables ###
            ########################

            ### layers that process the observation into the initial hidden state
            with tf.variable_scope('obs_inference_vars'):
                obs_weights = []
                obs_biases = []

                curr_layer = obs_dim
                obs_to_hidden_layers = self._obs_hidden_layers + [self._rnn_state_dim]
                if is_lstm:
                    obs_to_hidden_layers[-1] *= 2
                for i, next_layer in enumerate(obs_to_hidden_layers):
                    obs_weights.append(tf.get_variable('w_obs_hidden_{0}'.format(i), [curr_layer, next_layer],
                                                       initializer=tf.contrib.layers.xavier_initializer()))
                    obs_biases.append(tf.get_variable('b_obs_hidden_{0}'.format(i), [next_layer],
                                                      initializer=tf.constant_initializer(0.)))
                    curr_layer = next_layer

            ### layers that process each action at each time step
            with tf.variable_scope('action_inference_vars'):
                action_weights = []
                action_biases = []

                curr_layer = action_dim
                for i, next_layer in enumerate(self._action_hidden_layers + [self._rnn_state_dim]):
                    action_weights.append(tf.get_variable('w_action_hidden_{0}'.format(i), [curr_layer, next_layer],
                                                          initializer=tf.contrib.layers.xavier_initializer()))
                    action_biases.append(tf.get_variable('b_action_hidden_{0}'.format(i), [next_layer],
                                                         initializer=tf.constant_initializer(0.)))
                    curr_layer = next_layer

            ### layers that process each rnn output to produce reward
            with tf.variable_scope('reward_inference_vars'):
                reward_weights = []
                reward_biases = []

                curr_layer = self._rnn_state_dim
                for i, next_layer in enumerate(self._reward_hidden_layers + [1]):
                    reward_weights.append(tf.get_variable('w_reward_hidden_{0}'.format(i), [curr_layer, next_layer],
                                                          initializer=tf.contrib.layers.xavier_initializer()))
                    reward_biases.append(tf.get_variable('b_reward_hidden_{0}'.format(i), [next_layer],
                                                         initializer=tf.constant_initializer(0.)))
                    curr_layer = next_layer

            ### weight decay
            for v in obs_weights + action_weights + reward_weights:
                tf.add_to_collection('weight_decays', 0.5 * tf.reduce_mean(v * v))

            ####################
            ### Create graph ###
            ####################

            ### obs --> internal state
            with tf.name_scope('obs_to_istate'):
                layer = tf_obs
                for obs_weight, obs_bias in zip(obs_weights, obs_biases):
                    layer = self._activation(tf.add(tf.matmul(layer, obs_weight), obs_bias))
                istate = layer

            ### actions --> rnn input at each time step
            with tf.name_scope('actions_to_rnn_input'):
                tf_actions_list = tf.split(1, self._H, tf_actions)
                rnn_inputs = []
                for h in range(self._H):
                    layer = tf_actions_list[h]
                    for action_weight, action_bias in zip(action_weights, action_biases):
                        layer = self._activation(tf.add(tf.matmul(layer, action_weight), action_bias))
                    rnn_inputs.append(layer)
                rnn_inputs = tf.pack(rnn_inputs, 1)

            ### create rnn
            with tf.name_scope('rnn'):
                with tf.variable_scope('rnn_vars'):
                    if is_lstm:
                        rnn_cell = tf.nn.rnn_cell.BasicLSTMCell(self._rnn_state_dim,
                                                                state_is_tuple=False,
                                                                activation=self._rnn_activation)
                    else:
                        rnn_cell = tf.nn.rnn_cell.BasicRNNCell(self._rnn_state_dim, activation=self._rnn_activation)
                    rnn_outputs, rnn_states = tf.nn.dynamic_rnn(rnn_cell, rnn_inputs, initial_state=istate)

            ### internal states --> rewards
            with tf.name_scope('istates_to_rewards'):
                rewards = []
                for h in range(self._H):
                    layer = rnn_outputs[:, h, :]
                    for i, (reward_weight, reward_bias) in enumerate(zip(reward_weights, reward_biases)):
                        layer = tf.add(tf.matmul(layer, reward_weight), reward_bias)
                        if i < len(reward_weights) - 1:
                            layer = self._activation(layer)
                    rewards.append(layer)
                tf_rewards = tf.concat(1, rewards)

            tf_rewards = self._graph_preprocess_outputs(tf_rewards, d_preprocess)

        return tf_rewards

    @property
    def recurrent(self):
        return True
