import tensorflow as tf
import tensorflow.contrib.layers as layers

from rllab.misc.overrides import overrides
from rllab.core.serializable import Serializable
from sandbox.gkahn.rnn_critic.policies.policy import Policy
from sandbox.gkahn.rnn_critic.utils import tf_utils

class MultiactionCombinedcostMLPPolicy(Policy, Serializable):
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

        self._hidden_layers = list(hidden_layers)
        self._activation = eval(activation)
        self._concat_or_bilinear = concat_or_bilinear

        Policy.__init__(self, **kwargs)

        assert(self._N > 1)
        assert(self._H > 1)
        assert(self._N == self._H)
        assert(self._concat_or_bilinear == 'concat' or self._concat_or_bilinear == 'bilinear')

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
            tf_obs, tf_actions = self._graph_preprocess_inputs(tf_obs_ph, tf_actions_ph, d_preprocess)

            layer = layers.fully_connected(tf_obs, num_outputs=self._hidden_layers[0], activation_fn=self._activation,
                                           weights_regularizer=layers.l2_regularizer(1.))
            ### add actions after 1 hidden layer
            if self._concat_or_bilinear == 'concat':
                layer = tf.concat(1, [layer, tf_actions])
            elif self._concat_or_bilinear == 'bilinear':
                layer = tf.concat(1, (tf_utils.batch_outer_product_2d(layer, tf_actions), layer, tf_actions))
            else:
                raise Exception

            ### fully connected
            for num_outputs in self._hidden_layers[1:]:
                layer = layers.fully_connected(layer, num_outputs=num_outputs, activation_fn=self._activation,
                                               weights_regularizer=layers.l2_regularizer(1.))
            layer = layers.fully_connected(layer, num_outputs=output_dim, activation_fn=None,
                                           weights_regularizer=layers.l2_regularizer(1.))

            tf_rewards = self._graph_preprocess_outputs(layer, d_preprocess)


        return tf_rewards
