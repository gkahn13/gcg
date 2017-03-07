import numpy as np
import tensorflow as tf

from sandbox.rocky.tf.spaces.discrete import Discrete
from rllab.core.serializable import Serializable
from sandbox.gkahn.rnn_critic.policies.policy import RNNCriticPolicy

class RNNCriticDiscreteRNNPolicy(RNNCriticPolicy, Serializable):
    def __init__(self,
                 obs_hidden_layers,
                 reward_hidden_layers,
                 rnn_state_dim,
                 activation,
                 rnn_activation,
                 **kwargs):
        """
        :param obs_hidden_layers: list of layer sizes for obs --> rnn initial hidden states
        :param reward_hidden_layers: list of layer sizes for rnn outputs --> reward
        :param rnn_state_dim: dimension of the hidden state
        :param activation: str to be evaluated (e.g. 'tf.nn.relu')
        :param rnn_activation: str to be evaluated (e.g. 'tf.nn.relu')
        """
        assert(isinstance(kwargs.get('env_spec').action_space, Discrete))

        Serializable.quick_init(self, locals())

        self._obs_hidden_layers = list(obs_hidden_layers)
        self._reward_hidden_layers = list(reward_hidden_layers)
        self._rnn_state_dim = rnn_state_dim
        self._activation = eval(activation)
        self._rnn_activation = eval(rnn_activation)

        RNNCriticPolicy.__init__(self, **kwargs)

    ###########################
    ### TF graph operations ###
    ###########################

    def _graph_preprocess_from_placeholders(self):
        d_preprocess = dict()

        with tf.variable_scope('preprocess'):
            for name, dim in (('observations', self._env_spec.observation_space.flat_dim),
                              ('rewards', 1)):
                d_preprocess[name + '_mean_ph'] = tf.placeholder(tf.float32, shape=(1, dim), name=name + '_mean_ph')
                d_preprocess[name + '_mean_var'] = tf.get_variable(name + '_mean_var', shape=[1, dim],
                                                                   trainable=False, dtype=tf.float32,
                                                                   initializer=tf.constant_initializer(np.zeros((1, dim))))
                d_preprocess[name + '_mean_assign'] = tf.assign(d_preprocess[name + '_mean_var'],
                                                                d_preprocess[name + '_mean_ph'])

                d_preprocess[name + '_orth_ph'] = tf.placeholder(tf.float32, shape=(dim, dim), name=name + '_orth_ph')
                d_preprocess[name + '_orth_var'] = tf.get_variable(name + '_orth_var',
                                                                   shape=(dim, dim),
                                                                   trainable=False, dtype=tf.float32,
                                                                   initializer=tf.constant_initializer(np.eye(dim)))
                d_preprocess[name + '_orth_assign'] = tf.assign(d_preprocess[name + '_orth_var'],
                                                                d_preprocess[name + '_orth_ph'])

        return d_preprocess

    def _graph_preprocess_inputs(self, tf_obs_ph, tf_actions_ph, d_preprocess):
        ### whiten inputs
        tf_obs_whitened = tf.matmul(tf_obs_ph - d_preprocess['observations_mean_var'],
                                    d_preprocess['observations_orth_var'])

        num_obs = tf.shape(tf_obs_whitened)[0]
        num_action = tf.shape(tf_actions_ph)[0]

        def tf_repeat_2d(x, reps):
            """ Repeats x on axis=0 reps times """
            x_shape = tf.shape(x)
            x_repeat = tf.reshape(tf.tile(x, [1, reps]), (x_shape[0] * reps, x_shape[1]))
            x_repeat.set_shape((None, x.get_shape()[1]))
            return x_repeat

        ### replicate observation for each action
        def replicate_observation():
            tf_obs_whitened_rep = tf_repeat_2d(tf_obs_whitened, num_action // num_obs)
            # tf_obs_whitened_tiled = tf.tile(tf_obs_whitened, tf.pack([batch_size, 1]))
            # tf_obs_whitened_tiled.set_shape([None, tf_obs_whitened.get_shape()[1]])
            return tf_obs_whitened_rep

        # assumes num_action is a multiple of num_obs
        tf_obs_whitened_cond = tf.cond(tf.not_equal(num_obs, num_action), replicate_observation, lambda: tf_obs_whitened)

        return tf_obs_whitened_cond

    def _graph_preprocess_outputs(self, tf_rewards, d_preprocess):
        return (tf_rewards * d_preprocess['rewards_orth_var'][0, 0]) + d_preprocess['rewards_mean_var']

    def _graph_inference(self, tf_obs_ph, tf_actions_ph, d_preprocess):
        """
        - create a new RNNCell that contains len(actions) RNN cells in it
          and takes as input the current action, muxes it to get the output and next hidden state
        """
        input_shape = self._env_spec.observation_space.shape
        action_dim = self._env_spec.action_space.flat_dim
        assert (len(input_shape) == 1)  # TODO

        input_dim = np.prod(input_shape)

        with tf.name_scope('inference'):
            tf_obs = self._graph_preprocess_inputs(tf_obs_ph, tf_actions_ph, d_preprocess)

            ########################
            ### Create variables ###
            ########################

            ### layers that process the observation into the initial hidden state
            with tf.variable_scope('obs_inference_vars'):
                obs_weights = []
                obs_biases = []

                curr_layer = input_dim
                obs_to_hidden_layers = self._obs_hidden_layers + [self._rnn_state_dim]
                for i, next_layer in enumerate(obs_to_hidden_layers):
                    obs_weights.append(tf.get_variable('w_obs_hidden_{0}'.format(i), [curr_layer, next_layer],
                                                       initializer=tf.contrib.layers.xavier_initializer()))
                    obs_biases.append(tf.get_variable('b_obs_hidden_{0}'.format(i), [next_layer],
                                                      initializer=tf.constant_initializer(0.)))
                    curr_layer = next_layer

            ### layers that process each rnn output to produce reward
            with tf.variable_scope('reward_inference_vars'):
                reward_weights = []
                reward_biases = []

                curr_layer = self._rnn_state_dim
                for i, next_layer in enumerate(self._reward_hidden_layers + [1]):
                    reward_weights.append(
                        tf.get_variable('w_reward_hidden_{0}'.format(i), [curr_layer, next_layer],
                                        initializer=tf.contrib.layers.xavier_initializer()))
                    reward_biases.append(tf.get_variable('b_reward_hidden_{0}'.format(i), [next_layer],
                                                         initializer=tf.constant_initializer(0.)))
                    curr_layer = next_layer

            ### weight decay
            for v in obs_weights + reward_weights:
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

            ### create rnn
            with tf.name_scope('rnn'):
                with tf.variable_scope('rnn_vars'):
                    rnn_cell = tf.nn.rnn_cell.BasicRNNCell(self._rnn_state_dim, activation=self._rnn_activation)
                    rnn_inputs = tf.reshape(tf_actions_ph, (-1, self._H, action_dim))
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

    ################
    ### Training ###
    ################

    def update_preprocess(self, preprocess_stats):
        obs_mean, obs_orth, rewards_mean, rewards_orth = \
            preprocess_stats['observations_mean'], \
            preprocess_stats['observations_orth'], \
            preprocess_stats['rewards_mean'], \
            preprocess_stats['rewards_orth']

        self._tf_sess.run([
            self._d_preprocess['observations_mean_assign'],
            # self._d_preprocess['observations_orth_assign'],
            self._d_preprocess['rewards_mean_assign'],
            # self._d_preprocess['rewards_orth_assign']
        ],
            feed_dict={
                self._d_preprocess['observations_mean_ph']: obs_mean,
                self._d_preprocess['rewards_mean_ph']: np.expand_dims(rewards_mean * self._H, 0),
            # b/c sum of rewards
            })
