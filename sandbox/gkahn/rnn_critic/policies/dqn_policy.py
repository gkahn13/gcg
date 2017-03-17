import tensorflow as tf
import tensorflow.contrib.layers as layers

from rllab.misc.overrides import overrides
from rllab.core.serializable import Serializable
from sandbox.gkahn.rnn_critic.policies.policy import Policy
from sandbox.gkahn.rnn_critic.utils import tf_utils

class DQNPolicy(Policy, Serializable):
    def __init__(self,
                 hidden_layers,
                 activation,
                 concat_or_bilinear,
                 conv_hidden_layers=None,
                 conv_kernels=None,
                 conv_strides=None,
                 conv_activation=None,
                 **kwargs):
        """
        :param hidden_layers: list of layer sizes
        :param activation: str to be evaluated (e.g. 'tf.nn.relu')
        :param concat_or_bilinear: concat initial state or bilinear initial state
        """
        Serializable.quick_init(self, locals())

        self._hidden_layers = list(hidden_layers)
        self._activation = eval(activation)
        self._concat_or_bilinear = concat_or_bilinear
        self._use_conv = (conv_hidden_layers is not None) and (conv_kernels is not None) and \
                         (conv_strides is not None) and (conv_activation is not None)
        if self._use_conv:
            self._conv_hidden_layers = list(conv_hidden_layers)
            self._conv_kernels = list(conv_kernels)
            self._conv_strides = list(conv_strides)
            self._conv_activation = eval(conv_activation)

        Policy.__init__(self, **kwargs)

        assert(self._H == 1)
        assert(self._cost_type == 'combined')
        assert(self._concat_or_bilinear == 'concat' or self._concat_or_bilinear == 'bilinear')
        assert(len(self._hidden_layers) >= 1)

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
        output_dim = self.N_output

        with tf.name_scope('inference'):
            if self._use_conv:
                tf_obs, tf_actions = self._graph_preprocess_inputs(tf_obs_ph, tf_actions_ph, d_preprocess)

                obs_shape = list(self._env_spec.observation_space.shape)
                obs_shape[-1] = self._obs_history_len
                layer = tf.reshape(tf_obs, [-1] + list(obs_shape))
                for num_outputs, kernel_size, stride in zip(self._conv_hidden_layers,
                                                            self._conv_kernels,
                                                            self._conv_strides):
                    layer = layers.convolution2d(layer,
                                                 num_outputs=num_outputs,
                                                 kernel_size=kernel_size,
                                                 stride=stride,
                                                 activation_fn=self._conv_activation)
                layer = layers.flatten(layer)

                ### fully connected
                layer = layers.fully_connected(layer, num_outputs=self._hidden_layers[0], activation_fn=self._activation,
                                               weights_regularizer=layers.l2_regularizer(1.))
                ### add actions after 1 hidden layer
                if self._concat_or_bilinear == 'concat':
                    layer = tf.concat(1, [layer, tf_actions])
                elif self._concat_or_bilinear == 'bilinear':
                    layer = tf.concat(1, (tf_utils.batch_outer_product_2d(layer, tf_actions), layer, tf_actions))
                else:
                    raise Exception

                for num_outputs in self._hidden_layers[1:]:
                    layer = layers.fully_connected(layer, num_outputs=num_outputs, activation_fn=self._activation,
                                                   weights_regularizer=layers.l2_regularizer(1.))
                layer = layers.fully_connected(layer, num_outputs=output_dim, activation_fn=None,
                                               weights_regularizer=layers.l2_regularizer(1.))
                tf_rewards = self._graph_preprocess_outputs(layer, d_preprocess)

            else:
                if self._concat_or_bilinear == 'concat':
                    tf_obs, tf_actions = self._graph_preprocess_inputs(tf_obs_ph, tf_actions_ph, d_preprocess)
                    layer = tf.concat(1, [tf_obs, tf_actions])
                elif self._concat_or_bilinear == 'bilinear':
                    tf_obs, tf_actions = self._graph_preprocess_inputs(tf_obs_ph, tf_actions_ph, d_preprocess)
                    layer = tf.concat(1, (tf_utils.batch_outer_product_2d(tf_obs, tf_actions), tf_obs, tf_actions))
                else:
                    raise Exception

                ### fully connected
                for num_outputs in self._hidden_layers:
                    layer = layers.fully_connected(layer, num_outputs=num_outputs, activation_fn=self._activation,
                                                   weights_regularizer=layers.l2_regularizer(1.))
                layer = layers.fully_connected(layer, num_outputs=output_dim, activation_fn=None,
                                               weights_regularizer=layers.l2_regularizer(1.))

                tf_rewards = self._graph_preprocess_outputs(layer, d_preprocess)

            # import numpy as np
            # num_vars = np.sum([np.prod(v.get_shape()) for v in tf.trainable_variables()])
            # print('num_vars: {0}'.format(num_vars))
            # import IPython; IPython.embed()

        return tf_rewards

    ################
    ### Training ###
    ################

    @overrides
    def update_preprocess(self, preprocess_stats):
        """ Only observations """
        obs_mean, obs_orth = \
            preprocess_stats['observations_mean'], \
            preprocess_stats['observations_orth'], \
        # we assume if obs is im, the obs orth is the diagonal of the covariance
        self._tf_sess.run([
            self._d_preprocess['observations_mean_assign'],
            self._d_preprocess['observations_orth_assign'],
        ],
            feed_dict={
                self._d_preprocess['observations_mean_ph']: obs_mean,
                self._d_preprocess['observations_orth_ph']: obs_orth,
            })
