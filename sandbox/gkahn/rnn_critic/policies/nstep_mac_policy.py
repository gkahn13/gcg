import numpy as np
import tensorflow as tf

from rllab.core.serializable import Serializable
from rllab.misc.overrides import overrides

from sandbox.gkahn.rnn_critic.policies.mac_policy import MACPolicy

class NstepMACPolicy(MACPolicy, Serializable):
    def __init__(self, **kwargs):
        Serializable.quick_init(self, locals())

        MACPolicy.__init__(self, **kwargs)

        assert(self._H == 1)

    ###########################
    ### TF graph operations ###
    ###########################

    @overrides
    def _graph_inference(self, tf_obs_lowd, tf_actions_ph, values_softmax, tf_preprocess, add_reg=True):
        """
        :param tf_obs_lowd: [batch_size, self._rnn_state_dim]
        :param tf_actions_ph: [batch_size, H, action_dim]
        :param values_softmax: string
        :param tf_preprocess:
        :return: tf_values: [batch_size, H]
        """
        batch_size = tf.shape(tf_obs_lowd)[0]
        H = tf_actions_ph.get_shape()[1].value
        N = self._N
        assert (tf_obs_lowd.get_shape()[1].value == (2 * self._rnn_state_dim if self._use_lstm else self._rnn_state_dim))
        tf.assert_equal(tf.shape(tf_obs_lowd)[0], tf.shape(tf_actions_ph)[0])

        if self._use_lstm:
            istate = tf.nn.rnn_cell.LSTMStateTuple(*tf.split(1, 2, tf_obs_lowd))  # so state_is_tuple=True
        else:
            istate = tf_obs_lowd


        action = tf_actions_ph[:, 0, :]
        _, _, val, _ = \
            self._graph_inference_step(0, N, batch_size, istate, action, values_softmax, add_reg=add_reg)

        tf_values = tf.tile(val, (1, self._N))
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
        tf_nstep_rewards = None
        tf_nstep_values = None

        assert (tf_values.get_shape()[1].value == N)

        return tf_values, tf_values_softmax, tf_nstep_rewards, tf_nstep_values
