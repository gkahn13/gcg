import tensorflow as tf
import tensorflow.contrib.layers as layers

from rllab.core.serializable import Serializable
from rllab.misc.overrides import overrides

from sandbox.rocky.tf.spaces.discrete import Discrete
from sandbox.gkahn.rnn_critic.policies.mac_policy import MACPolicy
from sandbox.gkahn.rnn_critic.utils import tf_utils

class DQNPolicy(MACPolicy, Serializable):
    def __init__(self, **kwargs):
        Serializable.quick_init(self, locals())

        self._hidden_layers = kwargs.get('hidden_layers')

        kwargs['obs_hidden_layers'] = []
        kwargs['action_hidden_layers'] = []
        kwargs['reward_hidden_layers'] = []
        kwargs['rnn_state_dim'] = 0
        kwargs['use_lstm'] = False
        kwargs['use_bilinear'] = False
        kwargs['rnn_activation'] = 'None'
        MACPolicy.__init__(self, **kwargs)

        assert(isinstance(self._env_spec.action_space, Discrete))

        assert(self._H == 1)
        assert(self._N == 1)

    ###########################
    ### TF graph operations ###
    ###########################

    @overrides
    def _graph_obs_to_lowd(self, tf_obs_ph, tf_preprocess):
        with tf.name_scope('obs_to_lowd'):
            ### whiten observations
            obs_dim = self._env_spec.observation_space.flat_dim
            if tf_obs_ph.dtype != tf.float32:
                tf_obs_ph = tf.cast(tf_obs_ph, tf.float32)
            tf_obs_ph = tf.reshape(tf_obs_ph, (-1, self._obs_history_len * obs_dim))
            if self._obs_is_im:
                tf_obs_whitened = tf.mul(tf_obs_ph -
                                         tf.tile(tf_preprocess['observations_mean_var'], (1, self._obs_history_len)),
                                         tf.tile(tf_preprocess['observations_orth_var'], (self._obs_history_len,)))
            else:
                tf_obs_whitened = tf.matmul(tf_obs_ph -
                                            tf.tile(tf_preprocess['observations_mean_var'], (1, self._obs_history_len)),
                                            tf_utils.block_diagonal(
                                                [tf_preprocess['observations_orth_var']] * self._obs_history_len))
            tf_obs_whitened = tf.reshape(tf_obs_whitened, (-1, self._obs_history_len, obs_dim))
            # tf_obs_whitened = tf_obs_ph # TODO

            ### obs --> lower dimensional space
            if self._use_conv:
                obs_shape = [self._obs_history_len] + list(self._env_spec.observation_space.shape)[:2]
                layer = tf.transpose(tf.reshape(tf_obs_whitened, [-1] + list(obs_shape)), perm=(0, 2, 3, 1))
                for i, (num_outputs, kernel_size, stride) in enumerate(zip(self._conv_hidden_layers,
                                                                           self._conv_kernels,
                                                                           self._conv_strides)):
                    layer = layers.convolution2d(layer,
                                                 num_outputs=num_outputs,
                                                 kernel_size=kernel_size,
                                                 stride=stride,
                                                 activation_fn=self._conv_activation,
                                                 scope='obs_to_lowd_conv{0}'.format(i))
                layer = layers.flatten(layer)
            else:
                layer = layers.flatten(tf_obs_whitened)

            ### obs --> internal state
            final_dim = self._env_spec.action_space.flat_dim
            for i, num_outputs in enumerate(self._hidden_layers + [final_dim]):
                layer = layers.fully_connected(layer, num_outputs=num_outputs, activation_fn=self._activation,
                                               weights_regularizer=layers.l2_regularizer(1.),
                                               scope='obs_to_istate_fc{0}'.format(i))
            tf_obs_lowd = layer

        return tf_obs_lowd

    @overrides
    def _graph_inference(self, tf_obs_lowd, tf_actions_ph, tf_preprocess):
        """
        :param tf_obs_lowd: [batch_size, self._rnn_state_dim]
        :param tf_actions_ph: [batch_size, H, action_dim]
        :param tf_preprocess:
        :return: tf_values: [batch_size, H]
        """
        H = tf_actions_ph.get_shape()[1].value
        assert(H == 1)
        tf.assert_equal(tf.shape(tf_obs_lowd)[0], tf.shape(tf_actions_ph)[0])

        with tf.name_scope('inference'):
            tf_values = tf.expand_dims(tf.reduce_sum(tf_obs_lowd * tf_actions_ph[:, 0, :], reduction_indices=1), 1)

        assert(tf_values.get_shape()[1].value == H)

        return tf_values

    @overrides
    def _graph_get_action(self, tf_obs_lowd, get_action_params, tf_preprocess):
        """
        :param tf_obs_lowd: [batch_size, rnn_state_dim]
        :param H: max horizon to choose action over
        :param get_action_params: how to select actions
        :param tf_preprocess:
        :return: tf_get_action [batch_size, action_dim], tf_get_action_value [batch_size]
        """
        num_obs = tf.shape(tf_obs_lowd)[0]
        action_dim = self._env_spec.action_space.flat_dim

        tf_values_all = tf_obs_lowd
        tf_get_action = tf.one_hot(tf.argmax(tf_values_all, 1), depth=action_dim)
        tf_get_action_value = tf.reduce_max(tf_values_all, 1)

        ### check shapes
        tf.assert_equal(tf.shape(tf_get_action)[0], num_obs)
        tf.assert_equal(tf.shape(tf_get_action_value)[0], num_obs)
        assert(tf_get_action.get_shape()[1].value == action_dim)

        return tf_get_action, tf_get_action_value

    ################
    ### Training ###
    ################

    @overrides
    def update_preprocess(self, preprocess_stats):
        for key in ('actions_mean', 'actions_orth',
                    'rewards_mean', 'rewards_orth'):
            assert(self._preprocess_params[key] is False)
        MACPolicy.update_preprocess(self, preprocess_stats)
