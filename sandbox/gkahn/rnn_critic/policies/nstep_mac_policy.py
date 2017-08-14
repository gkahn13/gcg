import numpy as np
import tensorflow as tf

from rllab.core.serializable import Serializable
from rllab.misc.overrides import overrides

from sandbox.gkahn.rnn_critic.policies.mac_policy import MACPolicy
from sandbox.gkahn.rnn_critic.tf import networks

class NstepMACPolicy(MACPolicy, Serializable):
    def __init__(self, **kwargs):
        Serializable.quick_init(self, locals())

        MACPolicy.__init__(self, **kwargs)

        assert(self._H == 1)

    ###########################
    ### TF graph operations ###
    ###########################

    @overrides
    def _graph_inference(self, tf_obs_lowd, tf_actions_ph, values_softmax, tf_preprocess, is_training, add_reg=True):
        """
        :param tf_obs_lowd: [batch_size, self._rnn_state_dim]
        :param tf_actions_ph: [batch_size, H, action_dim]
        :param values_softmax: string
        :param tf_preprocess:
        :return: tf_values: [batch_size, H]
        """
        batch_size = tf.shape(tf_obs_lowd)[0]
        H = tf_actions_ph.get_shape()[1].value
        assert(H == self._H)
        N = self._N
        # tf.assert_equal(tf.shape(tf_obs_lowd)[0], tf.shape(tf_actions_ph)[0])

        action_dim = tf_actions_ph.get_shape()[2].value
        actions = tf.reshape(tf_actions_ph, (-1, action_dim))
        rnn_inputs, _ = networks.fcnn(actions, self._action_graph, is_training=is_training, scope='fcnn_actions')
        rnn_inputs = tf.reshape(rnn_inputs, (-1, H, self._action_graph['output_dim']))

        rnn_outputs, _ = networks.rnn(rnn_inputs, self._rnn_graph, initial_state=tf_obs_lowd)
        rnn_output_dim = rnn_outputs.get_shape()[2].value
        rnn_outputs = tf.reshape(rnn_outputs, (-1, rnn_output_dim))

        self._output_graph.update({'output_dim': 1})
        tf_values, _ = networks.fcnn(rnn_outputs, self._output_graph, is_training=is_training, scope='fcnn_values')
        tf_values = tf.tile(tf.reshape(tf_values, (-1, H)), (1, N))

        tf_nstep_rewards = None
        tf_nstep_values = None

        with tf.name_scope('nstep_lambdas'):
            if values_softmax['type'] == 'final':
                tf_values_softmax = tf.zeros([batch_size, N])
                tf_values_softmax[:, -1] = 1.
            elif values_softmax['type'] == 'mean':
                tf_values_softmax = (1. / float(H)) * tf.ones([batch_size, N])
            elif values_softmax['type'] == 'exponential':
                lam = values_softmax['exponential']['lambda']
                tf_values_softmaxes = []
                for h in range(N):
                    if h == N - 1:
                        tf_values_softmaxes.append(np.power(lam, h) * tf.ones([batch_size]))
                    else:
                        tf_values_softmaxes.append(tf_values_softmax = (1 - lam) * np.power(lam, h) * tf.ones([batch_size]))
                tf_values_softmax = tf.stack(tf_values_softmaxes, 1)
            else:
                raise NotImplementedError

        assert(tf_values.get_shape()[1].value == N)

        return tf_values, tf_values_softmax, tf_nstep_rewards, tf_nstep_values