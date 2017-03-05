import numpy as np
import tensorflow as tf

from sandbox.rocky.tf.spaces.discrete import Discrete
from rllab.core.serializable import Serializable
from sandbox.gkahn.rnn_critic.policies.policy import RNNCriticPolicy

class RNNCriticDiscreteMLPPolicy(RNNCriticPolicy, Serializable):
    def __init__(self,
                 hidden_layers,
                 activation,
                 **kwargs):
        """
        :param hidden_layers: list of layer sizes
        :param activation: str to be evaluated (e.g. 'tf.nn.relu')
        """
        assert(isinstance(kwargs.get('env_spec').action_space, Discrete))

        Serializable.quick_init(self, locals())

        self._hidden_layers = list(hidden_layers)
        self._activation = eval(activation)

        RNNCriticPolicy.__init__(self, **kwargs)

    ###########################
    ### TF graph operations ###
    ###########################

    def _graph_preprocess_from_placeholders(self):
        d_preprocess = dict()

        with tf.variable_scope('preprocess'):
            for name, dim in (('observations', self._env_spec.observation_space.flat_dim),
                              ('rewards', 1)):
                d_preprocess[name+'_mean_ph'] = tf.placeholder(tf.float32, shape=(1, dim), name=name+'_mean_ph')
                d_preprocess[name+'_mean_var'] = tf.get_variable(name+'_mean_var', shape=[1, dim],
                                                                 trainable=False, dtype=tf.float32,
                                                                 initializer=tf.constant_initializer(np.zeros((1, dim))))
                d_preprocess[name+'_mean_assign'] = tf.assign(d_preprocess[name+'_mean_var'],
                                                              d_preprocess[name+'_mean_ph'])

                d_preprocess[name+'_orth_ph'] = tf.placeholder(tf.float32, shape=(dim, dim), name=name+'_orth_ph')
                d_preprocess[name+'_orth_var'] = tf.get_variable(name+'_orth_var',
                                                                 shape=(dim, dim),
                                                                 trainable=False, dtype=tf.float32,
                                                                 initializer=tf.constant_initializer(np.eye(dim)))
                d_preprocess[name+'_orth_assign'] = tf.assign(d_preprocess[name+'_orth_var'],
                                                              d_preprocess[name+'_orth_ph'])

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

    def _graph_inference(self, tf_obs_ph, tf_actions_ph, d_preprocess):
        input_shape = self._env_spec.observation_space.shape
        action_dim = self._env_spec.action_space.flat_dim
        assert(len(input_shape) == 1) # TODO

        input_dim = np.prod(input_shape)
        output_dim = np.power(action_dim, self._H)

        with tf.name_scope('inference'):

            ### weights
            with tf.variable_scope('inference_vars'):
                weights = []
                biases = []

                curr_layer = input_dim
                for i, next_layer in enumerate(self._hidden_layers + [output_dim]):
                    weights.append(tf.get_variable('w_hidden_{0}'.format(i), [curr_layer, next_layer],
                                                   initializer=tf.contrib.layers.xavier_initializer()))
                    biases.append(tf.get_variable('b_hidden_{0}'.format(i), [next_layer],
                                                  initializer=tf.constant_initializer(0.)))
                    curr_layer = next_layer

            ### weight decays
            for v in weights:
                tf.add_to_collection('weight_decays', 0.5 * tf.reduce_mean(tf.square(v)))

            ### fully connected
            tf_obs = self._graph_preprocess_inputs(tf_obs_ph, tf_actions_ph, d_preprocess)
            layer = tf_obs
            for i, (weight, bias) in enumerate(zip(weights, biases)):
                with tf.name_scope('hidden_{0}'.format(i)):
                    layer = tf.add(tf.matmul(layer, weight), bias)
                    if i < len(weights) - 1:
                        layer = self._activation(layer)
            rewards_all = layer

            ### compute relevant rewards idxs
            with tf.name_scope('reward_idxs'):
                # action_dim = 4
                # H = 2
                # tf_actions_ph
                # [0, 0, 0, 1, 0, 0, 0, 1]
                tf_b = tf.tile(tf.expand_dims(tf.range(action_dim), 0), (1, self._H)) * tf.cast(tf_actions_ph, tf.int32)
                # [0, 0, 0, 3, 0, 0, 0, 3]
                tf_c = tf_b * np.power(action_dim, tf.constant(np.repeat(np.arange(self._H), (action_dim,)), dtype=tf.int32))
                # [0, 0, 0, 3, 0, 0, 0, 12]
                reward_idxs = tf.reduce_sum(tf_c, axis=1)
                # [15]

            ### extract relevant sum of rewards
            with tf.name_scope('gather_rewards'):
                reward_idxs_flat = reward_idxs + tf.range(tf.shape(reward_idxs)[0]) * np.power(action_dim, self._H)
                rewards = tf.expand_dims(tf.gather(tf.reshape(rewards_all, [-1]), reward_idxs_flat), 1)

            tf_rewards = self._graph_preprocess_outputs(rewards, d_preprocess)

            tf.assert_equal(tf.shape(tf_rewards)[0], tf.shape(tf_actions_ph)[0])

        return tf_rewards

    def _graph_cost(self, tf_rewards_ph, tf_rewards, tf_target_rewards, tf_target_mask_ph):
        # for training, len(tf_obs_ph) == len(tf_actions_ph)
        # but len(tf_actions_target_ph) == N * len(tf_action_ph),
        # so need to be selective about what to take the max over
        tf_target_values = tf_target_rewards # b/c network outputs the sum anyways
        batch_size = tf.shape(tf_rewards_ph)[0]
        tf_target_values_flat = tf.reshape(tf_target_values, (batch_size, -1))
        tf_target_values_max = tf.reduce_max(tf_target_values_flat, reduction_indices=1)

        tf_values_ph = self._graph_calculate_values(tf_rewards_ph)
        tf_values = tf_rewards # b/c network outputs the sum anyways

        mse = tf.reduce_mean(tf.square(tf_values_ph +
                                       tf_target_mask_ph * np.power(self._gamma, self._H)*tf_target_values_max -
                                       tf_values))
        weight_decay = self._weight_decay * tf.add_n(tf.get_collection('weight_decays'))
        cost = mse + weight_decay
        return cost, mse

    ################
    ### Training ###
    ################

    def update_preprocess(self, preprocess_stats):
        obs_mean, obs_orth,  rewards_mean, rewards_orth = \
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
                self._d_preprocess['rewards_mean_ph']: np.expand_dims(rewards_mean * self._H, 0), # b/c sum of rewards
            })
