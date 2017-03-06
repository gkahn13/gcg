import numpy as np
import tensorflow as tf

from sandbox.rocky.tf.spaces.discrete import Discrete
from rllab.core.serializable import Serializable
from sandbox.gkahn.rnn_critic.policies.policy import RNNCriticPolicy

class RNNCriticDiscreteRNNPolicy(RNNCriticPolicy, Serializable):
    def __init__(self,
                 obs_hidden_layers,
                 reward_hidden_layers,
                 rnn_state_dim,
                 activation,
                 rnn_activation,
                 **kwargs):
        """
        :param obs_hidden_layers: list of layer sizes for obs --> rnn initial hidden states
        :param reward_hidden_layers: list of layer sizes for rnn outputs --> reward
        :param rnn_state_dim: dimension of the hidden state
        :param activation: str to be evaluated (e.g. 'tf.nn.relu')
        :param rnn_activation: str to be evaluated (e.g. 'tf.nn.relu')
        """
        assert(isinstance(kwargs.get('env_spec').action_space, Discrete))

        Serializable.quick_init(self, locals())

        self._obs_hidden_layers = list(obs_hidden_layers)
        self._reward_hidden_layers = list(reward_hidden_layers)
        self._rnn_state_dim = rnn_state_dim
        self._activation = activation
        self._rnn_activation = rnn_activation

        RNNCriticPolicy.__init__(self, **kwargs)

    ###########################
    ### TF graph operations ###
    ###########################

    def _graph_preprocess_from_placeholders(self):
        d_preprocess = dict()

        with tf.variable_scope('preprocess'):
            for name, dim in (('observations', self._env_spec.observation_space.flat_dim),
                              ('rewards', 1)):
                d_preprocess[name + '_mean_ph'] = tf.placeholder(tf.float32, shape=(1, dim), name=name + '_mean_ph')
                d_preprocess[name + '_mean_var'] = tf.get_variable(name + '_mean_var', shape=[1, dim],
                                                                   trainable=False, dtype=tf.float32,
                                                                   initializer=tf.constant_initializer(np.zeros((1, dim))))
                d_preprocess[name + '_mean_assign'] = tf.assign(d_preprocess[name + '_mean_var'],
                                                                d_preprocess[name + '_mean_ph'])

                d_preprocess[name + '_orth_ph'] = tf.placeholder(tf.float32, shape=(dim, dim), name=name + '_orth_ph')
                d_preprocess[name + '_orth_var'] = tf.get_variable(name + '_orth_var',
                                                                   shape=(dim, dim),
                                                                   trainable=False, dtype=tf.float32,
                                                                   initializer=tf.constant_initializer(np.eye(dim)))
                d_preprocess[name + '_orth_assign'] = tf.assign(d_preprocess[name + '_orth_var'],
                                                                d_preprocess[name + '_orth_ph'])

        return d_preprocess


    def _graph_preprocess_inputs(self, tf_obs_ph, d_preprocess):
        ### whiten inputs
        tf_obs_whitened = tf.matmul(tf_obs_ph - d_preprocess['observations_mean_var'],
                                    d_preprocess['observations_orth_var'])

        return tf_obs_whitened


    def _graph_preprocess_outputs(self, tf_rewards, d_preprocess):
        return (tf_rewards * d_preprocess['rewards_orth_var'][0, 0]) + d_preprocess['rewards_mean_var']

    def _graph_inference(self, tf_obs_ph, tf_actions_ph, d_preprocess):
        """
        - have current rnn hidden state
        - use mux to select which rnn cell to use

        OR

        - use actions to select rnn cells
        - eval using these rnn cells

        OR

        - create a new RNNCell that contains len(actions) RNN cells in it
          and takes as input the current action, muxes it to get the output and next hidden state
        """
        pass # TODO