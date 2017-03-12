import tensorflow as tf
import tensorflow.contrib.layers as layers

from rllab.misc.overrides import overrides
from rllab.core.serializable import Serializable
from sandbox.gkahn.rnn_critic.policies.policy import Policy

class MultiactionCombinedcostMLPPolicy(Policy, Serializable):
    def __init__(self,
                 hidden_layers,
                 activation,
                 **kwargs):
        """
        :param hidden_layers: list of layer sizes
        :param activation: str to be evaluated (e.g. 'tf.nn.relu')
        """
        Serializable.quick_init(self, locals())

        self._hidden_layers = list(hidden_layers)
        self._activation = eval(activation)

        Policy.__init__(self, **kwargs)

        assert(self._N > 1)
        assert(self._H > 1)
        assert(self._N == self._H)

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
        output_dim = self.N_output

        with tf.name_scope('inference'):
            tf_obs, tf_actions = self._graph_preprocess_inputs(tf_obs_ph, tf_actions_ph, d_preprocess)

            layer = tf.concat(1, [tf_obs, tf_actions])

            ### fully connected
            for num_outputs in self._hidden_layers:
                layer = layers.fully_connected(layer, num_outputs=num_outputs, activation_fn=self._activation,
                                               weights_regularizer=layers.l2_regularizer(1.))
            layer = layers.fully_connected(layer, num_outputs=output_dim, activation_fn=None,
                                           weights_regularizer=layers.l2_regularizer(1.))

            tf_rewards = self._graph_preprocess_outputs(layer, d_preprocess)

        return tf_rewards
