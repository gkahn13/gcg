import os
from collections import defaultdict

import numpy as np
from sklearn.utils.extmath import cartesian

import tensorflow as tf
# import tensorflow.contrib.layers as layers

from rllab.core.serializable import Serializable
import rllab.misc.logger as logger
from rllab.misc import ext

from sandbox.rocky.tf.spaces.discrete import Discrete
from sandbox.gkahn.tf.policies.base import Policy as TfPolicy
from sandbox.gkahn.tf.core import xplatform
from sandbox.gkahn.rnn_critic.utils import schedules, tf_utils
from sandbox.gkahn.rnn_critic.tf.mulint_rnn_cell import BasicMulintRNNCell, BasicMulintLSTMCell

### exploration strategies
from sandbox.gkahn.rnn_critic.exploration_strategies.epsilon_greedy_strategy import EpsilonGreedyStrategy
from sandbox.gkahn.rnn_critic.exploration_strategies.gaussian_strategy import GaussianStrategy

class MACPolicy(TfPolicy, Serializable):
    def __init__(self, **kwargs):
        Serializable.quick_init(self, locals())

        ### environment
        self._env_spec = kwargs['env_spec']

        ### model horizons
        self._N = kwargs['N'] # number of returns to use (N-step)
        self._H = kwargs['H'] # action planning horizon for training
        self._gamma = kwargs['gamma'] # reward decay
        self._obs_history_len = kwargs['obs_history_len'] # how many previous observations to use

        ### model architecture
        self._obs_hidden_layers = list(kwargs['obs_hidden_layers'])
        self._action_hidden_layers = list(kwargs['action_hidden_layers'])
        self._reward_hidden_layers = list(kwargs['reward_hidden_layers'])
        self._value_hidden_layers = list(kwargs['value_hidden_layers'])
        self._lambda_hidden_layers = list(kwargs['lambda_hidden_layers'])
        self._rnn_state_dim = kwargs['rnn_state_dim']
        self._use_lstm = kwargs['use_lstm']
        self._use_bilinear = kwargs['use_bilinear']
        self._share_weights = kwargs['share_weights']
        self._activation = eval(kwargs['activation'])
        self._rnn_activation = eval(kwargs['rnn_activation'])
        self._use_conv = ('conv_hidden_layers' in kwargs) and ('conv_kernels' in kwargs) and \
                         ('conv_strides' in kwargs) and ('conv_activation' in kwargs)
        if self._use_conv:
            self._conv_hidden_layers = list(kwargs['conv_hidden_layers'])
            self._conv_kernels = list(kwargs['conv_kernels'])
            self._conv_strides = list(kwargs['conv_strides'])
            self._conv_activation = eval(kwargs['conv_activation'])

        ### target network
        self._values_softmax = kwargs['values_softmax'] # which value horizons to train over
        self._use_target = kwargs['use_target']
        self._separate_target_params = kwargs['separate_target_params']
        self._separate_mses = kwargs['separate_mses']

        ### training
        self._weight_decay = kwargs['weight_decay']
        self._lr_schedule = schedules.PiecewiseSchedule(**kwargs['lr_schedule'])
        self._grad_clip_norm = kwargs['grad_clip_norm']
        self._preprocess_params = kwargs['preprocess']
        self._gpu_device = kwargs['gpu_device']
        self._gpu_frac = kwargs['gpu_frac']

        ### action selection and exploration
        self._get_action_test = kwargs['get_action_test']
        self._get_action_target = kwargs['get_action_target']
        self._retrace_lambda = kwargs['retrace_lambda']
        es_params = kwargs['exploration_strategy']
        es_class = es_params['class']
        ESClass = eval(es_class)
        self._exploration_strategy = ESClass(env_spec=self._env_spec, **es_params[es_class])

        ### setup the model
        self._tf_debug = dict()
        self._tf_dict = self._graph_setup()

        ### logging
        self._log_stats = defaultdict(list)

        TfPolicy.__init__(self, self._env_spec, sess=self._tf_dict['sess'])

        assert((self._N == 1 and self._H == 1) or
               (self._N > 1 and self._H == 1) or
               (self._N > 1 and self._H > 1 and self._N == self._H))

    ##################
    ### Properties ###
    ##################

    @property
    def N(self):
        return self._N

    @property
    def gamma(self):
        return self._gamma

    @property
    def session(self):
        return self._tf_dict['sess']

    @property
    def _obs_is_im(self):
        return len(self._env_spec.observation_space.shape) > 1

    @property
    def obs_history_len(self):
        return self._obs_history_len

    ###########################
    ### TF graph operations ###
    ###########################

    @staticmethod
    def create_session_and_graph(gpu_device=None, gpu_frac=None):
        if gpu_device is None:
            gpu_device = 0
        if gpu_frac is None:
            gpu_frac = 0.3

        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_device)
        tf_graph = tf.Graph()
        if len(str(gpu_device)) > 0:
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_frac)
            config = tf.ConfigProto(gpu_options=gpu_options,
                                    log_device_placement=False,
                                    allow_soft_placement=True,
                                    inter_op_parallelism_threads=1,
                                    intra_op_parallelism_threads=1)
        else:
            config = tf.ConfigProto(
                device_count={'GPU': 0},
                log_device_placement=False,
                allow_soft_placement=True,
                inter_op_parallelism_threads=1,
                intra_op_parallelism_threads=1
            )
        tf_sess = tf.Session(graph=tf_graph, config=config)
        return tf_sess, tf_graph

    def _graph_input_output_placeholders(self):
        obs_shape = self._env_spec.observation_space.shape
        obs_dtype = tf.uint8 if len(obs_shape) > 1 else tf.float32
        obs_dim = self._env_spec.observation_space.flat_dim
        action_dim = self._env_spec.action_space.flat_dim

        with tf.variable_scope('input_output_placeholders'):
            ### policy inputs
            tf_obs_ph = tf.placeholder(obs_dtype, [None, self._obs_history_len, obs_dim], name='tf_obs_ph')
            tf_actions_ph = tf.placeholder(tf.float32, [None, self._N + 1, action_dim], name='tf_actions_ph')
            tf_dones_ph = tf.placeholder(tf.bool, [None, self._N], name='tf_dones_ph')
            ### policy outputs
            tf_rewards_ph = tf.placeholder(tf.float32, [None, self._N], name='tf_rewards_ph')
            ### target inputs
            tf_obs_target_ph = tf.placeholder(obs_dtype, [None, self._N + self._obs_history_len - 0, obs_dim], name='tf_obs_target_ph')
            ### policy exploration
            tf_explore_train_ph = tf.placeholder(tf.float32, [None, self._N + 1], name='tf_explore_train')
            tf_explore_test_ph = tf.placeholder(tf.float32, [None], name='tf_explore_test')
            ### importance sampling
            tf_logprob_prior_ph = tf.placeholder(tf.float32, [None, self._N], name='tf_logprob_prior_ph')

        return tf_obs_ph, tf_actions_ph, tf_dones_ph, tf_rewards_ph, tf_obs_target_ph, \
               tf_explore_train_ph, tf_explore_test_ph, tf_logprob_prior_ph

    def _graph_preprocess_placeholders(self):
        tf_preprocess = dict()

        with tf.variable_scope('preprocess'):
            for name, dim, diag_orth in (('observations', self._env_spec.observation_space.flat_dim, self._obs_is_im),
                                         ('actions', self._env_spec.action_space.flat_dim, False),
                                         ('rewards', 1, False)):
                tf_preprocess[name+'_mean_ph'] = tf.placeholder(tf.float32, shape=(1, dim), name=name+'_mean_ph')
                tf_preprocess[name+'_mean_var'] = tf.get_variable(name+'_mean_var', shape=[1, dim],
                                                                  trainable=False, dtype=tf.float32,
                                                                  initializer=tf.constant_initializer(np.zeros((1, dim))))
                tf_preprocess[name+'_mean_assign'] = tf.assign(tf_preprocess[name+'_mean_var'],
                                                               tf_preprocess[name+'_mean_ph'])

                if diag_orth:
                    tf_preprocess[name + '_orth_ph'] = tf.placeholder(tf.float32, shape=(dim,), name=name + '_orth_ph')
                    tf_preprocess[name + '_orth_var'] = tf.get_variable(name + '_orth_var',
                                                                        shape=(dim,),
                                                                        trainable=False, dtype=tf.float32,
                                                                        initializer=tf.constant_initializer(np.ones(dim)))
                else:
                    tf_preprocess[name + '_orth_ph'] = tf.placeholder(tf.float32, shape=(dim, dim), name=name + '_orth_ph')
                    tf_preprocess[name + '_orth_var'] = tf.get_variable(name + '_orth_var',
                                                                        shape=(dim, dim),
                                                                        trainable=False, dtype=tf.float32,
                                                                        initializer=tf.constant_initializer(np.eye(dim)))
                tf_preprocess[name+'_orth_assign'] = tf.assign(tf_preprocess[name+'_orth_var'],
                                                               tf_preprocess[name+'_orth_ph'])

        return tf_preprocess

    def _graph_obs_to_lowd(self, tf_obs_ph, tf_preprocess, add_reg=True):
        import tensorflow.contrib.layers as layers

        with tf.name_scope('obs_to_lowd'):
            ### whiten observations
            obs_dim = self._env_spec.observation_space.flat_dim
            if tf_obs_ph.dtype != tf.float32:
                tf_obs_ph = tf.cast(tf_obs_ph, tf.float32)
            tf_obs_ph = tf.reshape(tf_obs_ph, (-1, self._obs_history_len * obs_dim))
            if self._obs_is_im:
                tf_obs_whitened = tf.mul(tf_obs_ph -
                                         tf.tile(tf_preprocess['observations_mean_var'], (1, self._obs_history_len)),
                                         tf.tile(tf_preprocess['observations_orth_var'], (self._obs_history_len,)))
            else:
                tf_obs_whitened = tf.matmul(tf_obs_ph -
                                            tf.tile(tf_preprocess['observations_mean_var'], (1, self._obs_history_len)),
                                            tf_utils.block_diagonal(
                                                [tf_preprocess['observations_orth_var']] * self._obs_history_len))
            tf_obs_whitened = tf.reshape(tf_obs_whitened, (-1, self._obs_history_len, obs_dim))
            # tf_obs_whitened = tf_obs_ph # TODO

            ### obs --> lower dimensional space
            if self._use_conv:
                obs_shape = [self._obs_history_len] + list(self._env_spec.observation_space.shape)[:2]
                layer = tf.transpose(tf.reshape(tf_obs_whitened, [-1] + list(obs_shape)), perm=(0, 2, 3, 1))
                for i, (num_outputs, kernel_size, stride) in enumerate(zip(self._conv_hidden_layers,
                                                                           self._conv_kernels,
                                                                           self._conv_strides)):
                    layer = layers.convolution2d(layer,
                                                 num_outputs=num_outputs,
                                                 kernel_size=kernel_size,
                                                 stride=stride,
                                                 activation_fn=self._conv_activation,
                                                 scope='obs_to_lowd_conv{0}'.format(i))
                layer = layers.flatten(layer)
            else:
                layer = layers.flatten(tf_obs_whitened)

            ### obs --> internal state
            final_dim = self._rnn_state_dim if not self._use_lstm else 2 * self._rnn_state_dim
            for i, num_outputs in enumerate(self._obs_hidden_layers + [final_dim]):
                layer = layers.fully_connected(layer, num_outputs=num_outputs, activation_fn=self._activation,
                                               weights_regularizer=layers.l2_regularizer(1.) if add_reg else None,
                                               scope='obs_to_istate_fc{0}'.format(i))
            tf_obs_lowd = layer

        return tf_obs_lowd

    def _graph_inference(self, tf_obs_lowd, tf_actions_ph, values_softmax, tf_preprocess, add_reg=True, pad_inputs=True):
        """
        :param tf_obs_lowd: [batch_size, self._rnn_state_dim]
        :param tf_actions_ph: [batch_size, H, action_dim]
        :param values_softmax: string
        :param tf_preprocess:
        :param pad_inputs: if True, will be N inputs, otherwise will be H inputs
        :return: tf_values: [batch_size, H]
        """
        H = tf_actions_ph.get_shape()[1].value
        N = self._N if pad_inputs else H
        assert(tf_obs_lowd.get_shape()[1].value == (2 * self._rnn_state_dim if self._use_lstm else self._rnn_state_dim))
        tf.assert_equal(tf.shape(tf_obs_lowd)[0], tf.shape(tf_actions_ph)[0])

        if self._use_lstm:
            istate = tf.nn.rnn_cell.LSTMStateTuple(*tf.split(1, 2, tf_obs_lowd))  # so state_is_tuple=True
        else:
            istate = tf_obs_lowd

        tf_nstep_rewards = []
        tf_nstep_values = []
        tf_nstep_values_softmax = []
        for h in range(N):
            action = tf_actions_ph[:, h, :] if h < H else None

            next_istate, rew, val, vsoftmax = \
                self._graph_inference_step(h, N, istate, action, values_softmax, add_reg=add_reg)
            if self._share_weights:
                tf.get_variable_scope().reuse_variables()

            istate = next_istate
            tf_nstep_rewards.append(rew)
            tf_nstep_values.append(val)
            tf_nstep_values_softmax.append(vsoftmax)

        tf_values = tf.concat(1, [self._graph_calculate_value(n, tf_nstep_rewards, tf_nstep_values) for n in range(N)])
        tf_values_softmax = tf.pack(tf_nstep_values_softmax, 1)

        assert(tf_values.get_shape()[1].value == N)

        return tf_values, tf_values_softmax

    def _graph_inference_OLD(self, tf_obs_lowd, tf_actions_ph, values_softmax, tf_preprocess, add_reg=True, pad_inputs=True):
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
        assert(tf_obs_lowd.get_shape()[1].value == (2 * self._rnn_state_dim if self._use_lstm else self._rnn_state_dim))
        tf.assert_equal(tf.shape(tf_obs_lowd)[0], tf.shape(tf_actions_ph)[0])
        action_dim = self.env_spec.action_space.flat_dim

        with tf.name_scope('inference'):
            ### internal state
            istate = tf_obs_lowd

            ### preprocess actions
            # if isinstance(self._env_spec.action_space, Discrete):
            #     tf_actions = tf_actions_ph
            # else:
            #     tf_actions = tf.reshape(tf_actions_ph, (-1, H * action_dim))
            #     tf_actions = tf.matmul(tf_actions -
            #                            tf.tile(tf_preprocess['actions_mean_var'], (1, H)),
            #                            tf_utils.block_diagonal([tf_preprocess['actions_orth_var']] * H))
            #     tf_actions = tf.reshape(tf_actions, (-1, H, action_dim))
            tf_actions = tf_actions_ph # TODO

            ### actions --> rnn input at each time step
            with tf.name_scope('actions_to_rnn_input'):
                rnn_inputs = []
                ### actions
                for h in range(H):
                    layer = tf_actions[:, h, :]

                    for i, num_outputs in enumerate(self._action_hidden_layers + [self._rnn_state_dim]):
                        layer = layers.fully_connected(layer, num_outputs=num_outputs, activation_fn=self._activation,
                                                       weights_regularizer=layers.l2_regularizer(1.) if add_reg else None,
                                                       scope='actions_i{0}'.format(i),
                                                       reuse=tf.get_variable_scope().reuse or (h > 0))
                    rnn_inputs.append(layer)
                ### pad with zeros
                for h in range(H, N):
                    rnn_inputs.append(tf.zeros([tf.shape(tf_actions_ph)[0], self._rnn_state_dim]))

                rnn_inputs = tf.pack(rnn_inputs, 1)

            ### create rnn
            with tf.name_scope('rnn'):
                with tf.variable_scope('rnn_vars'):
                    if self._use_lstm:
                        if self._use_bilinear:
                            rnn_cell = BasicMulintLSTMCell(self._rnn_state_dim,
                                                           state_is_tuple=True,
                                                           activation=self._rnn_activation)
                        else:
                            rnn_cell = tf.nn.rnn_cell.BasicLSTMCell(self._rnn_state_dim,
                                                                    state_is_tuple=True,
                                                                    activation=self._rnn_activation)
                        istate = tf.nn.rnn_cell.LSTMStateTuple(*tf.split(1, 2, istate))  # so state_is_tuple=True
                    else:
                        if self._use_bilinear:
                            rnn_cell = BasicMulintRNNCell(self._rnn_state_dim, activation=self._rnn_activation)
                        else:
                            rnn_cell = tf.nn.rnn_cell.BasicRNNCell(self._rnn_state_dim, activation=self._rnn_activation)
                    rnn_outputs, rnn_states = tf.nn.dynamic_rnn(rnn_cell, rnn_inputs, initial_state=istate)

            ### internal state --> nstep rewards
            with tf.name_scope('istates_to_nstep_rewards'):
                tf_nstep_rewards = []
                for n in range(N):
                    layer = rnn_outputs[:, n, :]
                    for i, num_outputs in enumerate(self._reward_hidden_layers + [1]):
                        activation = self._activation if i < len(self._reward_hidden_layers) else None
                        layer = layers.fully_connected(layer,
                                                       num_outputs=num_outputs,
                                                       activation_fn=activation,
                                                       weights_regularizer=layers.l2_regularizer(1.) if add_reg else None,
                                                       scope='rewards_i{0}'.format(i),
                                                       reuse=tf.get_variable_scope().reuse or (n > 0))
                    tf_nstep_rewards.append(layer)
                # shifted by one
                tf_nstep_rewards.pop()
                tf_nstep_rewards.insert(0, 0)

            ### internal state --> nstep values
            with tf.name_scope('istates_to_nstep_values'):
                tf_nstep_values = []
                for n in range(N):
                    layer = rnn_outputs[:, n, :]
                    for i, num_outputs in enumerate(self._value_hidden_layers + [1]):
                        activation = self._activation if i < len(self._value_hidden_layers) else None
                        layer = layers.fully_connected(layer,
                                                       num_outputs=num_outputs,
                                                       activation_fn=activation,
                                                       weights_regularizer=layers.l2_regularizer(1.) if add_reg else None,
                                                       scope='values_i{0}'.format(i),
                                                       reuse=tf.get_variable_scope().reuse or (n > 0))
                    tf_nstep_values.append(layer)

            ### nstep rewards + nstep values --> values
            with tf.name_scope('nstep_rewards_nstep_values_to_values'):
                tf_values_list = []
                for n in range(N):
                    tf_returns = tf_nstep_rewards[:n] + [tf_nstep_values[n]]
                    tf_values_list.append(
                        np.sum([np.power(self._gamma, i) * tf_return for i, tf_return in enumerate(tf_returns)]))
                tf_values = tf.concat(1, tf_values_list)

            ### nstep lambdas --> values softmax and depth
            with tf.name_scope('nstep_lambdas_to_values_softmax_and_depth'):
                if values_softmax['type'] == 'final':
                    tf_values_softmax = tf.one_hot(N - 1, N) * tf.ones(tf.shape(tf_values))
                elif values_softmax['type'] == 'mean':
                    tf_values_softmax = (1. / float(N)) * tf.ones(tf.shape(tf_values))
                elif values_softmax['type'] == 'exponential':
                    lam = values_softmax['exponential']['lambda']
                    tf_values_softmax = np.array([np.power(lam, n) if n < N - 1 else (1 - lam) * np.power(lam, n)
                                                  for n in range(N)])
                else:
                    raise NotImplementedError

        assert(tf_values.get_shape()[1].value == N)

        return tf_values, tf_values_softmax

    def _graph_inference_step(self, n, N, istate, action, values_softmax, add_reg=True):
        """
        :param n: current step
        :param N: max step
        :param istate: current internal state
        :param action: if action is None, input zeros
        """
        import tensorflow.contrib.layers as layers
        batch_size = tf.shape(istate)[0]

        with tf.name_scope('inference_step'):
            ### action
            with tf.name_scope('action'):
                if action is None:
                    rnn_input = tf.zeros([batch_size, self._rnn_state_dim])
                else:
                    layer = action
                    for i, num_outputs in enumerate(self._action_hidden_layers + [self._rnn_state_dim]):
                        scope = 'actions_i{0}'.format(i) if self._share_weights else 'actions_n{0}_i{1}'.format(n, i)
                        layer = layers.fully_connected(layer, num_outputs=num_outputs, activation_fn=self._activation,
                                                       weights_regularizer=layers.l2_regularizer(1.) if add_reg else None,
                                                       scope=scope,
                                                       reuse=tf.get_variable_scope().reuse)
                    rnn_input = layer

            ### rnn
            with tf.name_scope('rnn'):
                scope = 'rnn' if self._share_weights else 'rnn_n{0}'.format(n)
                with tf.variable_scope(scope):
                    if self._use_lstm:
                        if self._use_bilinear:
                            rnn_cell = BasicMulintLSTMCell(self._rnn_state_dim,
                                                           state_is_tuple=True,
                                                           activation=self._rnn_activation)
                        else:
                            rnn_cell = tf.nn.rnn_cell.BasicLSTMCell(self._rnn_state_dim,
                                                                    state_is_tuple=True,
                                                                    activation=self._rnn_activation)
                    else:
                        if self._use_bilinear:
                            rnn_cell = BasicMulintRNNCell(self._rnn_state_dim, activation=self._rnn_activation)
                        else:
                            rnn_cell = tf.nn.rnn_cell.BasicRNNCell(self._rnn_state_dim, activation=self._rnn_activation)
                    tf_output, next_istate = rnn_cell(rnn_input, istate)

            ### rnn output --> nstep rewards
            with tf.name_scope('nstep_rewards'):
                layer = tf_output
                for i, num_outputs in enumerate(self._reward_hidden_layers + [1]):
                    activation = self._activation if i < len(self._reward_hidden_layers) else None
                    scope = 'rewards_i{0}'.format(i) if self._share_weights else 'rewards_n{0}_i{1}'.format(n, i)
                    layer = layers.fully_connected(layer,
                                                   num_outputs=num_outputs,
                                                   activation_fn=activation,
                                                   weights_regularizer=layers.l2_regularizer(1.) if add_reg else None,
                                                   scope=scope,
                                                   reuse=tf.get_variable_scope().reuse)
                tf_nstep_reward = layer

            ### rnn output --> nstep values
            with tf.name_scope('nstep_values'):
                layer = tf_output
                for i, num_outputs in enumerate(self._value_hidden_layers + [1]):
                    activation = self._activation if i < len(self._value_hidden_layers) else None
                    scope = 'values_i{0}'.format(i) if self._share_weights else 'values_n{0}_i{1}'.format(n, i)
                    layer = layers.fully_connected(layer,
                                                   num_outputs=num_outputs,
                                                   activation_fn=activation,
                                                   weights_regularizer=layers.l2_regularizer(1.) if add_reg else None,
                                                   scope=scope,
                                                   reuse=tf.get_variable_scope().reuse)
                tf_nstep_value = layer

            ### nstep lambdas --> values softmax and depth
            with tf.name_scope('nstep_lambdas'):
                if values_softmax['type'] == 'final':
                    if n == N - 1:
                        tf_values_softmax = tf.ones([batch_size])
                    else:
                        tf_values_softmax = tf.zeros([batch_size])
                elif values_softmax['type'] == 'mean':
                    tf_values_softmax = (1. / float(N)) * tf.ones([batch_size])
                elif values_softmax['type'] == 'exponential':
                    lam = values_softmax['exponential']['lambda']
                    if n == N - 1:
                        tf_values_softmax = np.power(lam, n) * tf.ones([batch_size])
                    else:
                        tf_values_softmax = (1 - lam) * np.power(lam, n) * tf.ones([batch_size])
                else:
                    raise NotImplementedError

        return next_istate, tf_nstep_reward, tf_nstep_value, tf_values_softmax

    def _graph_calculate_value(self, n, tf_nstep_rewards, tf_nstep_values):
        tf_returns = tf_nstep_rewards[:n] + [tf_nstep_values[n]]
        tf_value = np.sum([np.power(self._gamma, i) * tf_return for i, tf_return in enumerate(tf_returns)])
        return tf_value

    def _graph_generate_random_actions(self, K):
        action_dim = self._env_spec.action_space.flat_dim
        if isinstance(self._env_spec.action_space, Discrete):
            tf_actions = tf.one_hot(tf.random_uniform([K, 1], minval=0, maxval=action_dim, dtype=tf.int32),
                                    depth=action_dim,
                                    axis=2)
        else:
            action_lb = np.expand_dims(self._env_spec.action_space.low, 0)
            action_ub = np.expand_dims(self._env_spec.action_space.high, 0)
            tf_actions = (action_ub - action_lb) * tf.random_uniform([K, action_dim]) + action_lb

        return tf_actions

    def _graph_get_action(self, tf_obs_ph, get_action_params, scope_select, reuse_select, scope_eval, reuse_eval):
        """
        :param tf_obs_ph: [batch_size, obs_history_len, obs_dim]
        :param get_action_params: how to select actions
        :param scope_select: which scope to evaluate values (double Q-learning)
        :param scope_eval: which scope to select values (double Q-learning)
        :return: tf_get_action [batch_size, action_dim], tf_get_action_value [batch_size]
        """
        H = get_action_params['H']
        assert(H <= self._N)
        get_action_type = get_action_params['type']
        num_obs = tf.shape(tf_obs_ph)[0]
        action_dim = self._env_spec.action_space.flat_dim

        ### create actions
        if get_action_type == 'random':
            K = get_action_params[get_action_type]['K']
            if isinstance(self._env_spec.action_space, Discrete):
                tf_actions = tf.one_hot(tf.random_uniform([K, H], minval=0, maxval=action_dim, dtype=tf.int32),
                                        depth=action_dim,
                                        axis=2)
            else:
                action_lb = np.expand_dims(self._env_spec.action_space.low, 0)
                action_ub = np.expand_dims(self._env_spec.action_space.high, 0)
                tf_actions = (action_ub - action_lb) * tf.random_uniform([K, H, action_dim]) + action_lb
        elif get_action_type == 'lattice':
            assert(isinstance(self._env_spec.action_space, Discrete))
            indices = cartesian([np.arange(action_dim)] * H) + np.r_[0:action_dim * H:action_dim]
            actions = np.zeros((len(indices), action_dim * H))
            for i, one_hots in enumerate(indices):
                actions[i, one_hots] = 1
            actions = actions.reshape(len(indices), H, action_dim)
            K = len(actions)
            tf_actions = tf.constant(actions, dtype=tf.float32)
        else:
            raise NotImplementedError

        ### process to lowd
        with tf.variable_scope(scope_select, reuse=reuse_select):
            tf_preprocess_select = self._graph_preprocess_placeholders()
            tf_obs_lowd_select = self._graph_obs_to_lowd(tf_obs_ph, tf_preprocess_select)
        with tf.variable_scope(scope_eval, reuse=reuse_eval):
            tf_preprocess_eval = self._graph_preprocess_placeholders()
            tf_obs_lowd_eval = self._graph_obs_to_lowd(tf_obs_ph, tf_preprocess_eval)
        ### tile
        tf_actions = tf.tile(tf_actions, (num_obs, 1, 1))
        tf_obs_lowd_repeat_select = tf_utils.repeat_2d(tf_obs_lowd_select, K, 0)
        tf_obs_lowd_repeat_eval = tf_utils.repeat_2d(tf_obs_lowd_eval, K, 0)
        ### inference to get values
        with tf.variable_scope(scope_select, reuse=reuse_select):
            tf_values_all_select, tf_values_softmax_all_select = \
                self._graph_inference(tf_obs_lowd_repeat_select, tf_actions, get_action_params['values_softmax'],
                                      tf_preprocess_select, add_reg=False, pad_inputs=False)  # [num_obs*k, H]
        with tf.variable_scope(scope_eval, reuse=reuse_eval):
            tf_values_all_eval, tf_values_softmax_all_eval = \
                self._graph_inference(tf_obs_lowd_repeat_eval, tf_actions, get_action_params['values_softmax'],
                                      tf_preprocess_eval, add_reg=False, pad_inputs=False)  # [num_obs*k, H]
        ### get_action based on select (policy)
        tf_values_select = tf.reduce_sum(tf_values_all_select * tf_values_softmax_all_select, reduction_indices=1)  # [num_obs*K]
        tf_values_select = tf.reshape(tf_values_select, (num_obs, K))  # [num_obs, K]
        tf_values_argmax_select = tf.one_hot(tf.argmax(tf_values_select, 1), depth=K)  # [num_obs, K]
        tf_get_action = tf.reduce_sum(
            tf.tile(tf.expand_dims(tf_values_argmax_select, 2), (1, 1, action_dim)) *
            tf.reshape(tf_actions, (num_obs, K, H, action_dim))[:, :, 0, :],
            reduction_indices=1)  # [num_obs, action_dim]
        ### get_action_value based on eval (target)
        tf_values_eval = tf.reduce_sum(tf_values_all_eval * tf_values_softmax_all_eval, reduction_indices=1)  # [num_obs*K]
        tf_values_eval = tf.reshape(tf_values_eval, (num_obs, K))  # [num_obs, K]
        tf_get_action_value = tf.reduce_sum(tf_values_argmax_select * tf_values_eval, reduction_indices=1)

        ### check shapes
        tf.assert_equal(tf.shape(tf_get_action)[0], num_obs)
        tf.assert_equal(tf.shape(tf_get_action_value)[0], num_obs)
        assert(tf_get_action.get_shape()[1].value == action_dim)

        # if 'tf_actions' not in self._tf_debug:
        # self._tf_debug['tf_obs'] = tf_utils.repeat_2d(tf_obs_ph[:, 0, :], K, 0) # MATCH
        # self._tf_debug['tf_actions'] = tf_actions # MATCH
        # self._tf_debug['tf_values_all_select'] = tf_values_all_select
        # self._tf_debug['tf_values_softmax_all_select'] = tf_values_softmax_all_select
        # self._tf_debug['tf_values_select'] = tf_values_select
        # self._tf_debug['tf_values_argmax_select'] = tf_values_argmax_select
        # self._tf_debug['tf_get_action'] = tf_get_action
        # self._tf_debug['tf_values_eval'] = tf_values_eval
        # self._tf_debug['tf_get_action_value'] = tf_get_action_value

        return tf_get_action, tf_get_action_value

    def _graph_get_action_explore(self, tf_actions, tf_explore_ph):
        """
        :param tf_actions: [batch_size, action_dim]
        :param tf_explore_ph: [batch_size]
        :return:
        """
        action_dim = self._env_spec.action_space.flat_dim
        batch_size = tf.shape(tf_actions)[0]
        tf.assert_equal(batch_size, tf.shape(tf_explore_ph)[0])

        if isinstance(self._exploration_strategy, EpsilonGreedyStrategy):
            mask = tf.cast(tf.tile(tf.random_uniform([batch_size, 1]) < tf_explore_ph, (1, action_dim)), tf.float32)
            tf_actions_explore = (1 - mask) * tf_actions + mask * self._graph_generate_random_actions(batch_size)
        elif isinstance(self._exploration_strategy, GaussianStrategy):
            tf_actions_explore = tf.clip_by_value(tf_actions + tf.random_normal(tf.shape(tf_actions)) *
                                                  tf.tile(tf.expand_dims(tf_explore_ph, 1), (1, action_dim)),
                                                  self._env_spec.action_space.low,
                                                  self._env_spec.action_space.high)
        else:
            tf_actions_explore = tf_actions

        return tf_actions_explore

    def _graph_get_action_logprob(self, tf_actions, tf_actions_opt, tf_explore_ph):
        action_dim = self._env_spec.action_space.flat_dim
        batch_size = tf.shape(tf_actions)[0]
        tf.assert_equal(batch_size, tf.shape(tf_explore_ph)[0])

        if isinstance(self._exploration_strategy, EpsilonGreedyStrategy):
            is_same_actions = tf.cast(tf.argmax(tf_actions, 1) == tf.argmax(tf_actions_opt, 1), tf.float32)
            tf_actions_logprob = is_same_actions * tf.log(1 - tf_explore_ph) + \
                                 (1 - is_same_actions) * tf.log(tf_explore_ph / (max(1, action_dim - 1)))
        elif isinstance(self._exploration_strategy, GaussianStrategy):
            x = tf_actions
            mu = tf_actions_opt
            k = action_dim

            tf_actions_logprob = -0.5 * np.log(2 * np.pi)\
                                 -0.5 * k * tf.log(tf_explore_ph)\
                                 -0.5 * tf.reduce_sum((x - mu) * (x - mu), 1) / tf_explore_ph
        else:
            raise NotImplementedError

        return tf_actions_logprob

    def _graph_cost(self, tf_train_values, tf_train_values_softmax, tf_rewards_ph, tf_dones_ph,
                    tf_target_get_action_values, tf_logprob_curr, tf_logprob_prior):
        """
        :param tf_train_values: [None, self._N]
        :param tf_train_values_softmax: [None, self._N]
        :param tf_rewards_ph: [None, self._N]
        :param tf_dones_ph: [None, self._N]
        :param tf_target_get_action_values: [None, self._N]
        :param tf_logprob_curr: [None, self._N]
        :param tf_logprob_prior: [None, self._N]
        :return: tf_cost, tf_mse
        """
        batch_size = tf.shape(tf_dones_ph)[0]
        tf_dones = tf.cast(tf_dones_ph, tf.float32)

        if self._retrace_lambda is not None:
            relu_min = lambda x: -tf.nn.relu(-x)
            tf_logweights = tf.cumsum(relu_min(tf_logprob_curr - tf_logprob_prior), 1) + \
                            tf.cast(tf.range(self._N), tf.float32) * np.log(self._retrace_lambda)  # [batch_size, N]
            # now need to normalize across the columns (aka across the batch for each N-step)
            tf_weights_tilde = tf.exp(tf_logweights - tf.reduce_max(tf_logweights, 0))  # max for numerical stability
            tf_weights = tf_weights_tilde / tf.reduce_sum(tf_weights_tilde, 0)
        else:
            tf_weights = (1. / tf.cast(batch_size, tf.float32)) * tf.ones(tf.shape(tf_train_values_softmax))
        tf.assert_equal(tf.reduce_sum(tf_weights, 0), 1.)
        tf.assert_equal(tf.reduce_sum(tf_train_values_softmax, 1), 1.)

        tf_values_desired = []
        for n in range(self._N):
            tf_sum_rewards_n = tf.reduce_sum(np.power(self._gamma * np.ones(n + 1), np.arange(n + 1)) *
                                             tf_rewards_ph[:, :n + 1],
                                             reduction_indices=1)
            tf_target_values_n = (1 - tf_dones[:, n]) * np.power(self._gamma, n + 1) * tf_target_get_action_values[:, n]
            if self._separate_target_params:
                tf_target_values_n = tf.stop_gradient(tf_target_values_n)
            tf_values_desired.append(tf_sum_rewards_n + tf_target_values_n)
        tf_values_desired = tf.pack(tf_values_desired, 1)

        if self._separate_mses:
            tf_mse = tf.reduce_sum(tf_weights * tf_train_values_softmax * tf.square(tf_train_values - tf_values_desired))
        else:
            tf_mse = tf.reduce_sum(tf_weights[:, 0] *
                                   tf.reduce_mean(
                                       tf.square(tf_train_values -
                                                 tf.tile(
                                                     tf.reduce_sum(tf_train_values_softmax * tf_values_desired, axis=1, keep_dims=True),
                                                     (1, self._N))),
                                       1))


        ### weight decay
        if len(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)) > 0:
            tf_weight_decay = self._weight_decay * tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
        else:
            tf_weight_decay = 0
        tf_cost = tf_mse + tf_weight_decay

        return tf_cost, tf_mse

    def _graph_optimize(self, tf_cost, tf_policy_vars):
        tf_lr_ph = tf.placeholder(tf.float32, (), name="learning_rate")
        optimizer = tf.train.AdamOptimizer(learning_rate=tf_lr_ph, epsilon=1e-4)
        gradients = optimizer.compute_gradients(tf_cost, var_list=tf_policy_vars)
        for i, (grad, var) in enumerate(gradients):
            if grad is not None:
                gradients[i] = (tf.clip_by_norm(grad, self._grad_clip_norm), var)
        return optimizer.apply_gradients(gradients), tf_lr_ph

    def _graph_init_vars(self, tf_sess):
        tf_sess.run([xplatform.global_variables_initializer()])

    def _graph_setup(self):
        ### create session and graph
        tf_sess = tf.get_default_session()
        if tf_sess is None:
            tf_sess, tf_graph = MACPolicy.create_session_and_graph(gpu_device=self._gpu_device, gpu_frac=self._gpu_frac)
        tf_graph = tf_sess.graph

        with tf_sess.as_default(), tf_graph.as_default():
            if ext.get_seed() is not None:
                ext.set_seed(ext.get_seed())

            ### create input output placeholders
            tf_obs_ph, tf_actions_ph, tf_dones_ph, tf_rewards_ph, tf_obs_target_ph,\
                tf_explore_train_ph, tf_explore_test_ph, tf_logprob_prior_ph = self._graph_input_output_placeholders()

            ### policy
            policy_scope = 'policy'
            with tf.variable_scope(policy_scope):
                ### create preprocess placeholders
                tf_preprocess = self._graph_preprocess_placeholders()
                ### process obs to lowd
                tf_obs_lowd = self._graph_obs_to_lowd(tf_obs_ph, tf_preprocess)
                ### create training policy
                tf_train_values, tf_train_values_softmax = \
                    self._graph_inference(tf_obs_lowd, tf_actions_ph[:, :self._H, :],
                                          self._values_softmax, tf_preprocess)

            with tf.variable_scope(policy_scope, reuse=True):
                tf_train_values_test, tf_train_values_softmax_test = \
                    self._graph_inference(tf_obs_lowd, tf_actions_ph[:, :self._get_action_test['H'], :],
                                          self._values_softmax, tf_preprocess, pad_inputs=False)
                tf_get_value = tf.reduce_sum(tf_train_values_softmax_test * tf_train_values_test, reduction_indices=1)

            ### action selection
            tf_get_action, tf_get_action_value = self._graph_get_action(tf_obs_ph, self._get_action_test,
                                                                        policy_scope, True, policy_scope, True)
            ### exploration strategy and logprob
            tf_get_action_explore = self._graph_get_action_explore(tf_get_action, tf_explore_test_ph)
            tf_get_action_logprob = self._graph_get_action_logprob(tf_get_action_explore, tf_get_action, tf_explore_test_ph)

            ### get policy variables
            tf_policy_vars = sorted(tf.get_collection(xplatform.global_variables_collection_name(),
                                                      scope=policy_scope), key=lambda v: v.name)
            tf_trainable_policy_vars = sorted(tf.get_collection(xplatform.trainable_variables_collection_name(),
                                                                scope=policy_scope), key=lambda v: v.name)

            ### create target network
            if self._use_target:
                target_scope = 'target' if self._separate_target_params else 'policy'
                ### action selection
                tf_obs_target_ph_packed = tf.concat(0, [tf_obs_target_ph[:, h - self._obs_history_len:h, :]
                                                        for h in range(self._obs_history_len, self._obs_history_len + self._N + 1)])
                tf_target_get_action, tf_target_get_action_values = self._graph_get_action(tf_obs_target_ph_packed,
                                                                                           self._get_action_target,
                                                                                           scope_select=policy_scope,
                                                                                           reuse_select=True,
                                                                                           scope_eval=target_scope,
                                                                                           reuse_eval=(target_scope == policy_scope))

                ### logprob
                action_dim = self._env_spec.action_space.flat_dim
                tf_actions = tf.reshape(tf.reshape(tf_actions_ph, (-1, self._N + 1 * action_dim)), (-1, action_dim)) # [batch_size, N+1, action_dim] --> [batch_size * N+1
                tf_target_get_action_logprob = self._graph_get_action_logprob(tf_actions,
                                                                              tf_target_get_action,
                                                                              tf.reshape(tf_explore_train_ph, (-1,)))

                tf_target_get_action_values = tf.transpose(tf.reshape(tf_target_get_action_values, (self._N + 1, -1)))[:, 1:]
                tf_logprob_curr = tf.transpose(tf.reshape(tf_target_get_action_logprob, (self._N + 1, -1)))[:, :self._N]
            else:
                assert(self._retrace_lambda is None)
                tf_logprob_curr = None
                tf_target_get_action_values = tf.zeros([tf.shape(tf_train_values)[0], self._N])

            ### update target network
            if self._use_target and self._separate_target_params:
                tf_target_vars = sorted(tf.get_collection(xplatform.global_variables_collection_name(),
                                                          scope=target_scope), key=lambda v: v.name)
                assert(len(tf_policy_vars) == len(tf_target_vars))
                tf_update_target_fn = []
                for var, var_target in zip(tf_policy_vars, tf_target_vars):
                    assert(var.name.replace(policy_scope, '') == var_target.name.replace(target_scope, ''))
                    tf_update_target_fn.append(var_target.assign(var))
                tf_update_target_fn = tf.group(*tf_update_target_fn)
            else:
                tf_update_target_fn = None

            ### optimization
            tf_cost, tf_mse = self._graph_cost(tf_train_values, tf_train_values_softmax, tf_rewards_ph, tf_dones_ph,
                                               tf_target_get_action_values, tf_logprob_curr, tf_logprob_prior_ph)
            tf_opt, tf_lr_ph = self._graph_optimize(tf_cost, tf_trainable_policy_vars)

            ### initialize
            self._graph_init_vars(tf_sess)

        ### what to return
        return {
            'sess': tf_sess,
            'graph': tf_graph,
            'obs_ph': tf_obs_ph,
            'actions_ph': tf_actions_ph,
            'dones_ph': tf_dones_ph,
            'rewards_ph': tf_rewards_ph,
            'obs_target_ph': tf_obs_target_ph,
            'explore_train_ph': tf_explore_train_ph,
            'explore_test_ph': tf_explore_test_ph,
            'logprob_ph': tf_logprob_prior_ph,
            'preprocess': tf_preprocess,
            'get_value': tf_get_value,
            'get_action': tf_get_action,
            'get_action_explore': tf_get_action_explore,
            'get_action_value': tf_get_action_value,
            'get_action_logprob': tf_get_action_logprob,
            'update_target_fn': tf_update_target_fn,
            'cost': tf_cost,
            'mse': tf_mse,
            'opt': tf_opt,
            'lr_ph': tf_lr_ph,
        }

    ################
    ### Training ###
    ################

    def update_preprocess(self, preprocess_stats):
        obs_mean, obs_orth, actions_mean, actions_orth, rewards_mean, rewards_orth = \
            preprocess_stats['observations_mean'], \
            preprocess_stats['observations_orth'], \
            preprocess_stats['actions_mean'], \
            preprocess_stats['actions_orth'], \
            preprocess_stats['rewards_mean'], \
            preprocess_stats['rewards_orth']

        tf_assigns = []
        for key in ('observations_mean', 'observations_orth',
                    'actions_mean', 'actions_orth',
                    'rewards_mean', 'rewards_orth'):
            if self._preprocess_params[key]:
                tf_assigns.append(self._tf_dict['preprocess'][key + '_assign'])

        # we assume if obs is im, the obs orth is the diagonal of the covariance
        self._tf_dict['sess'].run(tf_assigns,
                          feed_dict={
                              self._tf_dict['preprocess']['observations_mean_ph']: obs_mean,
                              self._tf_dict['preprocess']['observations_orth_ph']: obs_orth,
                              self._tf_dict['preprocess']['actions_mean_ph']: actions_mean,
                              self._tf_dict['preprocess']['actions_orth_ph']: actions_orth,
                              # self._tf_dict['preprocess']['rewards_mean_ph']: (self._N / float(self.N_output)) *  # reweighting!
                              #                                        np.expand_dims(np.tile(rewards_mean, self.N_output),
                              #                                                       0),
                              # self._tf_dict['preprocess']['rewards_orth_ph']: (self._N / float(self.N_output)) *
                              #                                        scipy.linalg.block_diag(
                              #                                            *([rewards_orth] * self.N_output))
                          })

    def update_target(self):
        if self._use_target and self._separate_target_params:
            self._tf_dict['sess'].run(self._tf_dict['update_target_fn'])

    def train_step(self, step, steps, observations, actions, rewards, dones, logprobs, use_target):
        """
        :param steps: [batch_size, N+1]
        :param observations: [batch_size, N+1 + obs_history_len-1, obs_dim]
        :param actions: [batch_size, N+1, action_dim]
        :param rewards: [batch_size, N+1]
        :param dones: [batch_size, N+1]
        """
        feed_dict = {
            ### parameters
            self._tf_dict['lr_ph']: self._lr_schedule.value(step),
            self._tf_dict['explore_train_ph']: np.reshape([self._exploration_strategy.schedule.value(s)
                                                           for s in steps.ravel()], steps.shape),
            ### policy
            self._tf_dict['obs_ph']: observations[:, :self._obs_history_len, :],
            self._tf_dict['actions_ph']: actions,
            self._tf_dict['dones_ph']: np.logical_or(not use_target, dones[:, :self._N]),
            self._tf_dict['rewards_ph']: rewards[:, :self._N],
            self._tf_dict['logprob_ph']: logprobs[:, :self._N]
        }
        if self._use_target:
            feed_dict[self._tf_dict['obs_target_ph']] = observations

        cost, mse, _ = self._tf_dict['sess'].run([self._tf_dict['cost'],
                                                  self._tf_dict['mse'],
                                                  self._tf_dict['opt']],
                                                 feed_dict=feed_dict)
        assert(np.isfinite(cost))

        self._log_stats['Cost'].append(cost)
        self._log_stats['mse/cost'].append(mse / cost)

    ######################
    ### Policy methods ###
    ######################

    def get_action(self, step, observation, explore):
        chosen_actions, chosen_values, action_info = self.get_actions([step], [observation], explore=explore)
        return chosen_actions[0], chosen_values[0], action_info

    def get_actions(self, steps, observations, explore):
        d = {}
        # keys = [k for k in self._tf_debug.keys()]
        # vs = self._tf_dict['sess'].run([self._tf_debug[k] for k in keys],
        #                                 feed_dict={self._tf_dict['obs_ph']: observations})
        # for k, v in zip(keys, vs):
        #     d[k] = v

        # import IPython; IPython.embed()

        if explore:
            feed_dict = {
                self._tf_dict['obs_ph']: observations,
                self._tf_dict['explore_test_ph']: [self._exploration_strategy.schedule.value(t) for t in steps]

            }

            actions, values, logprobs = self._tf_dict['sess'].run([self._tf_dict['get_action_explore'],
                                                                  self._tf_dict['get_action_value'],
                                                                  self._tf_dict['get_action_logprob']],
                                                                  feed_dict=feed_dict)
        else:
            feed_dict = {
                self._tf_dict['obs_ph']: observations,
            }

            actions, values = self._tf_dict['sess'].run([self._tf_dict['get_action'],
                                                         self._tf_dict['get_action_value']],
                                                        feed_dict=feed_dict)
            logprobs = [np.nan] * len(steps)

        if isinstance(self._env_spec.action_space, Discrete):
            actions = [int(a.argmax()) for a in actions]

        return actions, values, logprobs, d

    def get_values(self, observations, actions):
        get_values = self._tf_dict['sess'].run(self._tf_dict['get_value'],
                                               feed_dict={self._tf_dict['obs_ph']: observations,
                                                          self._tf_dict['actions_ph']: actions})

        d = {}
        # keys = [k for k in self._tf_debug.keys() if 'train' in k]
        # values = self._tf_dict['sess'].run([self._tf_debug[k] for k in keys],
        #                                    feed_dict={self._tf_dict['obs_ph']: observations,
        #                                               self._tf_dict['actions_ph']: actions})
        # for k, v in zip(keys, values):
        #     d[k] = v

        return get_values, d

    @property
    def recurrent(self):
        return False

    def terminate(self):
        self._tf_dict['sess'].close()

    ######################
    ### Saving/loading ###
    ######################

    def get_params_internal(self, **tags):
        with self._tf_dict['graph'].as_default():
            return sorted(tf.get_collection(xplatform.global_variables_collection_name()), key=lambda v: v.name)

    ###############
    ### Logging ###
    ###############

    def log(self):
        for k in sorted(self._log_stats.keys()):
            if k == 'Depth':
                logger.record_tabular(k+'Mean', np.mean(self._log_stats[k]))
                logger.record_tabular(k+'Std', np.std(self._log_stats[k]))
            else:
                logger.record_tabular(k, np.mean(self._log_stats[k]))
        self._log_stats.clear()
