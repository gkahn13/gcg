import tensorflow as tf
import tensorflow.contrib.layers as layers

from rllab.misc.overrides import overrides
from rllab.core.serializable import Serializable
from sandbox.gkahn.rnn_critic.policies.dqn_policy import DQNPolicy
from sandbox.gkahn.rnn_critic.utils import tf_utils

class NstepDQNPolicy(DQNPolicy, Serializable):
    def __init__(self,
                 hidden_layers,
                 activation,
                 concat_or_bilinear,
                 **kwargs):
        """
        :param hidden_layers: list of layer sizes
        :param activation: str to be evaluated (e.g. 'tf.nn.relu')
        :param concat_or_bilinear: concat initial state or bilinear initial state
        """
        Serializable.quick_init(self, locals())

        DQNPolicy.__init__(self,
                           hidden_layers=hidden_layers,
                           activation=activation,
                           concat_or_bilinear=concat_or_bilinear,
                           **kwargs)

        assert(self._N > 1)
        assert(self._H == 1)
        assert(self._cost_type == 'combined')

    ##################
    ### Properties ###
    ##################

    @property
    def N_output(self):
        return self._N

    ###########################
    ### TF graph operations ###
    ###########################

    @overrides
    def _graph_inference(self, tf_obs_ph, tf_actions_ph, d_preprocess):
        with tf.name_scope('inference'):
            tf_obs, tf_actions = self._graph_preprocess_inputs(tf_obs_ph, tf_actions_ph, d_preprocess)
            # ensure same number of parameters as multiaction
            tf_actions_rep = tf.tile(tf_actions, (1, self._N))

            if self._concat_or_bilinear == 'concat':
                layer = tf.concat(1, [tf_obs, tf_actions_rep])
            elif self._concat_or_bilinear == 'bilinear':
                layer = tf.concat(1, (tf_utils.batch_outer_product_2d(tf_obs, tf_actions), tf_obs, tf_actions_rep))
            else:
                raise Exception

            ### fully connected
            for num_outputs in self._hidden_layers:
                layer = layers.fully_connected(layer, num_outputs=num_outputs, activation_fn=self._activation,
                                               weights_regularizer=layers.l2_regularizer(1.))
            # ensure same number of parameters as multiaction
            layer = layers.fully_connected(layer, num_outputs=self._N, activation_fn=None,
                                           weights_regularizer=layers.l2_regularizer(1.))

            # layer = tf.reduce_sum(layer, reduction_indices=1, keep_dims=True)

            tf_rewards = self._graph_preprocess_outputs(layer, d_preprocess)

        return tf_rewards

