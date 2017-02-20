import tensorflow as tf

from rllab.misc.overrides import overrides
from rllab.core.serializable import Serializable
from sandbox.gkahn.rnn_critic.policies.rnn_critic_policy import RNNCriticPolicy

class RNNCriticMLPPolicy(RNNCriticPolicy, Serializable):
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

        RNNCriticPolicy.__init__(self, **kwargs)

    @overrides
    def _graph_inference(self, tf_obs_ph, tf_actions_ph, d_preprocess):
        input_dim = self._env_spec.observation_space.flat_dim + self._env_spec.action_space.flat_dim * self._H
        output_dim = self._H

        with tf.name_scope('inference'):
            tf_obs, tf_actions = self._graph_preprocess_inputs(tf_obs_ph, tf_actions_ph, d_preprocess)

            input_layer = tf.concat(1, [tf_obs, tf_actions])

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
                tf.add_to_collection('weight_decays', 0.5 * tf.reduce_mean(v ** 2))

            ### fully connected
            layer = input_layer
            for i, (weight, bias) in enumerate(zip(weights, biases)):
                with tf.name_scope('hidden_{0}'.format(i)):
                    layer = tf.add(tf.matmul(layer, weight), bias)
                    if i < len(weights) - 1:
                        layer = self._activation(layer)
            tf_rewards = self._graph_preprocess_outputs(layer, d_preprocess)

        return tf_rewards

    @property
    def recurrent(self):
        return False
