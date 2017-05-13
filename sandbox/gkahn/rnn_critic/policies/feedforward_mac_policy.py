import tensorflow as tf
import numpy as np

from rllab.core.serializable import Serializable
from rllab.misc.overrides import overrides

from sandbox.rocky.tf.spaces.box import Box
from sandbox.gkahn.rnn_critic.policies.mac_policy import MACPolicy
from sandbox.gkahn.rnn_critic.utils import tf_utils

class FeedforwardMACPolicy(MACPolicy, Serializable):
    def __init__(self, **kwargs):
        Serializable.quick_init(self, locals())

        kwargs['action_hidden_layers'] = []
        kwargs['lambda_hidden_layers'] = []
        kwargs['rnn_state_dim'] = kwargs['obs_hidden_layers'].pop()
        kwargs['use_lstm'] = False
        kwargs['share_weights'] = False
        kwargs['rnn_activation'] = 'None'
        MACPolicy.__init__(self, **kwargs)

        assert(isinstance(self._env_spec.action_space, Box))

    ###########################
    ### TF graph operations ###
    ###########################

    @overrides
    def _graph_inference(self, tf_obs_lowd, tf_actions_ph, values_softmax, tf_preprocess, add_reg=True, pad_inputs=True):
        """
        :param tf_obs_lowd: [batch_size, self._rnn_state_dim]
        :param tf_actions_ph: [batch_size, H, action_dim]
        :param values_softmax: string
        :param tf_preprocess:
        :param pad_inputs: if True, will be N inputs, otherwise will be H inputs
        :return: tf_values: [batch_size, H]
        """
        import tensorflow.contrib.layers as layers

        H = tf_actions_ph.get_shape()[1].value
        N = self._N if pad_inputs else H
        batch_size = tf.shape(tf_obs_lowd)[0]
        action_dim = self._env_spec.action_space.flat_dim

        with tf.name_scope('inference'):
            layer = tf_obs_lowd
            tf_actions_flat = tf.reshape(tf_actions_ph, (batch_size, action_dim * H))
            if self._use_bilinear:
                pad_ones = tf.ones((batch_size, 1))
                layer = tf_utils.batch_outer_product_2d(tf.concat(1, [layer, pad_ones]),
                                                        tf.concat(1, [tf_actions_flat, pad_ones]))
            else:
                layer = tf.concat(1, [layer, tf_actions_flat])

            ### obs_lowd + action --> n-step reward
            for i, num_outputs in enumerate(self._reward_hidden_layers + [N]):
                activation = self._activation if i < len(self._reward_hidden_layers) else None
                layer = layers.fully_connected(layer, num_outputs=num_outputs, activation_fn=activation,
                                               weights_regularizer=layers.l2_regularizer(1.) if add_reg else None,
                                               scope='obs_lowd_and_action_to_reward_fc{0}'.format(i))
            tf_nstep_rewards = tf.split(1, N, layer)

            ### obs_lowd + action --> n-step value
            for i, num_outputs in enumerate(self._value_hidden_layers + [N]):
                activation = self._activation if i < len(self._value_hidden_layers) else None
                layer = layers.fully_connected(layer, num_outputs=num_outputs, activation_fn=activation,
                                               weights_regularizer=layers.l2_regularizer(1.) if add_reg else None,
                                               scope='obs_lowd_and_action_to_value_fc{0}'.format(i))
            tf_nstep_values = tf.split(1, N, layer)

            tf_values = tf.concat(1, [self._graph_calculate_value(n, tf_nstep_rewards, tf_nstep_values) for n in range(N)])

            if values_softmax['type'] == 'final':
                tf_values_softmax = tf.one_hot(N - 1, N) * tf.ones(tf.shape(tf_values))
            elif values_softmax['type'] == 'mean':
                tf_values_softmax = (1. / float(N)) * tf.ones(tf.shape(tf_values))
            elif values_softmax['type'] == 'exponential':
                lam = values_softmax['exponential']['lambda']
                lams = (1 - lam) * np.power(lam, np.arange(N - 1))
                lams = np.array(list(lams) + [np.power(lam, N - 1)])
                tf_values_softmax = lams * tf.ones(tf.shape(tf_values))
            else:
                raise NotImplementedError

        assert (tf_values.get_shape()[1].value == N)

        return tf_values, tf_values_softmax

    @overrides
    def _graph_inference_step(self, n, N, istate, action, values_softmax, add_reg=True):
        raise NotImplementedError
