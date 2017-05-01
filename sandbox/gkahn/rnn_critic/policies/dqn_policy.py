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
        kwargs['value_hidden_layers'] = []
        kwargs['lambda_hidden_layers'] = []
        kwargs['rnn_state_dim'] = 0
        kwargs['use_lstm'] = False
        kwargs['use_bilinear'] = False
        kwargs['rnn_activation'] = 'None'
        MACPolicy.__init__(self, **kwargs)

        assert(isinstance(self._env_spec.action_space, Discrete))

        assert(self._H == 1)

    ###########################
    ### TF graph operations ###
    ###########################

    @overrides
    def _graph_obs_to_lowd(self, tf_obs_ph, tf_preprocess, add_reg=True):
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
            for i, num_outputs in enumerate(self._hidden_layers):
                layer = layers.fully_connected(layer, num_outputs=num_outputs, activation_fn=self._activation,
                                               weights_regularizer=layers.l2_regularizer(1.) if add_reg else None,
                                               scope='obs_to_lowd_fc{0}'.format(i))
            layer = layers.fully_connected(layer, num_outputs=self._env_spec.action_space.flat_dim, activation_fn=None,
                                           weights_regularizer=layers.l2_regularizer(1.),
                                           scope='obs_to_lowd_fc_final')
            tf_obs_lowd = layer

        return tf_obs_lowd

    @overrides
    def _graph_inference(self, tf_obs_lowd, tf_actions_ph, values_softmax, tf_preprocess, add_reg=True, pad_inputs=True):
        """
        :param tf_obs_lowd: [batch_size, self._rnn_state_dim]
        :param tf_actions_ph: [batch_size, H, action_dim]
        :param values_softmax: string
        :param tf_preprocess:
        :return: tf_values: [batch_size, H]
        """
        H = tf_actions_ph.get_shape()[1].value
        N = self._N if pad_inputs else H
        assert(H == 1)
        tf.assert_equal(tf.shape(tf_obs_lowd)[0], tf.shape(tf_actions_ph)[0])

        with tf.name_scope('inference'):
            tf_values = tf.tile(tf.reduce_sum(tf_obs_lowd * tf_actions_ph[:, 0, :],
                                              reduction_indices=1, keep_dims=True),
                                (1, N))

            if values_softmax == 'final':
                tf_values_softmax = tf.one_hot(N - 1, N) * tf.ones(tf.shape(tf_values))
                tf_values_depth = (N - 1) * tf.ones([tf.shape(tf_values)[0]])
            elif values_softmax == 'mean':
                tf_values_softmax = (1. / float(N)) * tf.ones(tf.shape(tf_values))
                tf_values_depth = ((N - 1) / 2.) * tf.ones([tf.shape(tf_values)[0]])
            else:
                raise NotImplementedError

        assert(tf_values.get_shape()[1].value == N)

        return tf_values, tf_values_softmax, tf_values_depth

    ################
    ### Training ###
    ################

    @overrides
    def update_preprocess(self, preprocess_stats):
        for key in ('actions_mean', 'actions_orth',
                    'rewards_mean', 'rewards_orth'):
            assert(self._preprocess_params[key] is False)
        MACPolicy.update_preprocess(self, preprocess_stats)
