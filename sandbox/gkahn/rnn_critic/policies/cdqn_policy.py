import tensorflow as tf
import numpy as np

from rllab.core.serializable import Serializable
from rllab.misc.overrides import overrides

from sandbox.rocky.tf.spaces.box import Box
from sandbox.gkahn.rnn_critic.policies.mac_policy import MACPolicy
from sandbox.gkahn.rnn_critic.utils import tf_utils

class CDQNPolicy(MACPolicy, Serializable):
    def __init__(self, **kwargs):
        Serializable.quick_init(self, locals())

        kwargs['action_hidden_layers'] = []
        kwargs['reward_hidden_layers'] = []
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
    def _graph_inference(self, tf_obs_lowd, tf_actions_ph, values_softmax, tf_preprocess, is_training, add_reg=True, pad_inputs=True):
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
        assert(H == 1)
        N = self._N if pad_inputs else H
        batch_size = tf.shape(tf_obs_lowd)[0]

        with tf.name_scope('inference'):
            layer = tf_obs_lowd
            if self._use_bilinear:
                pad_ones = tf.ones((batch_size, 1))
                layer = tf_utils.batch_outer_product_2d(tf.concat([layer, pad_ones], 1),
                                                        tf.concat([tf_actions_ph[:, 0, :], pad_ones], 1))
            else:
                layer = tf.concat([layer, tf_actions_ph[:, 0, :]], 1)

            ### obs_lowd + action --> value
            for i, num_outputs in enumerate(self._value_hidden_layers + [1]):
                activation = self._activation if i < len(self._value_hidden_layers) else None
                layer = layers.fully_connected(layer, num_outputs=num_outputs, activation_fn=activation,
                                               weights_regularizer=layers.l2_regularizer(1.) if add_reg else None,
                                               scope='obs_lowd_and_action_to_value_fc{0}'.format(i))

            tf_values = tf.tile(layer, (1, N))

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

        return tf_values, tf_values_softmax, None, None

    @overrides
    def _graph_inference_step(self, n, N, istate, action, values_softmax, add_reg=True):
        raise NotImplementedError
