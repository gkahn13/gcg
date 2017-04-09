import time

import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers

from rllab.misc.overrides import overrides
from rllab.core.serializable import Serializable
from sandbox.gkahn.rnn_critic.policies.policy import Policy
from sandbox.rocky.tf.spaces.discrete import Discrete

from tensorflow.contrib.cudnn_rnn import CudnnRNNRelu, CudnnLSTM

from rllab.misc import ext

from sandbox.gkahn.tf.core import xplatform

class MultiactionCombinedcostRNNPolicy(Policy, Serializable):
    def __init__(self,
                 obs_hidden_layers,
                 action_hidden_layers,
                 reward_hidden_layers,
                 rnn_state_dim,
                 use_lstm,
                 activation,
                 rnn_activation,
                 batch_size,
                 conv_hidden_layers=None,
                 conv_kernels=None,
                 conv_strides=None,
                 conv_activation=None,
                 **kwargs):
        """
        :param obs_hidden_layers: layer sizes for preprocessing the observation
        :param action_hidden_layers: layer sizes for preprocessing the action
        :param reward_hidden_layers: layer sizes for processing the reward
        :param rnn_state_dim: dimension of the hidden state
        :param use_lstm: use lstm
        :param activation: string, e.g. 'tf.nn.relu'
        :param batch_size: for cudnn rnn
        """
        Serializable.quick_init(self, locals())

        self._obs_hidden_layers = list(obs_hidden_layers)
        self._action_hidden_layers = list(action_hidden_layers)
        self._reward_hidden_layers = list(reward_hidden_layers)
        self._rnn_state_dim = rnn_state_dim
        self._use_lstm = use_lstm
        self._activation = eval(activation)
        self._rnn_activation = eval(rnn_activation)
        self._batch_size = batch_size
        self._use_conv = (conv_hidden_layers is not None) and (conv_kernels is not None) and \
                         (conv_strides is not None) and (conv_activation is not None)
        if self._use_conv:
            self._conv_hidden_layers = list(conv_hidden_layers)
            self._conv_kernels = list(conv_kernels)
            self._conv_strides = list(conv_strides)
            self._conv_activation = eval(conv_activation)

        Policy.__init__(self, **kwargs)

        assert(self._N > 1)
        assert(self._H > 1)
        assert(self._N == self._H)

    ##################
    ### Properties ###
    ##################

    @property
    def N_output(self):
        return 1

    ###########################
    ### TF graph operations ###
    ###########################

    # @overrides
    # def _graph_inputs_outputs_from_placeholders(self):
    #     with tf.variable_scope('feed_input'):
    #         obs_shape = self._env_spec.observation_space.shape
    #         tf_obs_ph = tf.placeholder(tf.uint8 if len(obs_shape) > 1 else tf.float32,
    #                                    [self._batch_size, self._env_spec.observation_space.flat_dim * self._obs_history_len], name='tf_obs_ph')
    #         tf_actions_ph = tf.placeholder(tf.float32, [self._batch_size, self._env_spec.action_space.flat_dim * self._H], name='tf_actions_ph')
    #         tf_rewards_ph = tf.placeholder(tf.float32, [self._batch_size, self._N], name='tf_rewards_ph')
    #
    #     return tf_obs_ph, tf_actions_ph, tf_rewards_ph

    @overrides
    def _graph_inference(self, tf_obs_ph, tf_actions_ph, d_preprocess):
        with tf.name_scope('inference'):
            tf_obs, tf_actions = self._graph_preprocess_inputs(tf_obs_ph, tf_actions_ph, d_preprocess)

            ### obs --> lower dimensional space
            with tf.name_scope('obs_to_lowd'):
                if self._use_conv:
                    obs_shape = [self._obs_history_len] + list(self._env_spec.observation_space.shape)[:2]
                    layer = tf.transpose(tf.reshape(tf_obs, [-1] + list(obs_shape)), perm=(0, 2, 3, 1))
                    for num_outputs, kernel_size, stride in zip(self._conv_hidden_layers,
                                                                self._conv_kernels,
                                                                self._conv_strides):
                        layer = layers.convolution2d(layer,
                                                     num_outputs=num_outputs,
                                                     kernel_size=kernel_size,
                                                     stride=stride,
                                                     activation_fn=self._conv_activation)
                    layer = layers.flatten(layer)
                else:
                    layer = layers.flatten(tf_obs)

            ### obs --> internal state
            with tf.name_scope('obs_to_istate'):
                final_dim = self._rnn_state_dim if not self._use_lstm else 2 * self._rnn_state_dim
                for num_outputs in self._obs_hidden_layers + [final_dim]:
                    layer = layers.fully_connected(layer, num_outputs=num_outputs, activation_fn=self._activation,
                                                   weights_regularizer=layers.l2_regularizer(1.))
                istate = layer

            ### replicate istate if needed
            istate = self._graph_match_actions(istate, tf_actions)
            istate.set_shape([self._batch_size, final_dim]) # TODO

            ### actions --> rnn input at each time step
            with tf.name_scope('actions_to_rnn_input'):
                tf_actions_list = tf.split(1, self._N, tf_actions)
                rnn_inputs = []
                for h in range(self._N):
                    layer = tf_actions_list[h]

                    for i, num_outputs in enumerate(self._action_hidden_layers + [self._rnn_state_dim]):
                        layer = layers.fully_connected(layer, num_outputs=num_outputs, activation_fn=self._activation,
                                                       weights_regularizer=layers.l2_regularizer(1.),
                                                       scope='actions_i{0}'.format(i),
                                                       reuse=(h > 0))
                    rnn_inputs.append(layer)
                rnn_inputs = tf.pack(rnn_inputs, 1)

            rnn_inputs.set_shape([self._batch_size, self._N, self._rnn_state_dim]) # TODO

            ### create rnn
            with tf.name_scope('rnn'):
                with tf.variable_scope('rnn_vars'):
                    if self._use_lstm:
                        cudnn_rnn = CudnnLSTM(num_layers=self._N,
                                              num_units=self._rnn_state_dim,
                                              input_size=self._rnn_state_dim)

                        istate_h_0, istate_c_0 = tf.split(split_dim=1, num_split=2, value=istate)
                        # istate_h
                        istate_h_gt0 = tf.get_variable(name='istate_h_gt0',
                                                       initializer=tf.random_uniform([self._batch_size,
                                                                                      self._N-1,
                                                                                      self._rnn_state_dim]),
                                                       validate_shape=True)
                        istate_h = tf.transpose(tf.concat(1, [tf.expand_dims(istate_h_0, 1), istate_h_gt0]), (1, 0, 2))
                        # istate_h = tf.get_variable(name='istate_h',
                        #                            initializer=tf.random_uniform([self._N,
                        #                                                           self._batch_size,
                        #                                                           self._rnn_state_dim]),
                        #                            validate_shape=True)
                        # istate_c
                        istate_c_gt0 = tf.get_variable(name='istate_c_gt0',
                                                       initializer=tf.random_uniform([self._batch_size,
                                                                                      self._N-1,
                                                                                      self._rnn_state_dim]),
                                                       validate_shape=True)
                        istate_c = tf.transpose(tf.concat(1, [tf.expand_dims(istate_c_0, 1), istate_c_gt0]), (1, 0, 2))
                        # istate_c = tf.get_variable(name='istate_c',
                        #                            initializer=tf.random_uniform([self._N,
                        #                                                           self._batch_size,
                        #                                                           self._rnn_state_dim]),
                        #                            validate_shape=True)
                        # cudnn params
                        params_size = cudnn_rnn.params_size()
                        cudnn_rnn_params = tf.get_variable(name='cudnn_rnn_params',
                                                            # shape=cudnn_rnn.params_size(),
                                                            initializer=tf.random_uniform([params_size]),
                                                            validate_shape=False)
                        params_size_eval = tf.get_default_session().run(params_size)
                        cudnn_rnn_params.set_shape([params_size_eval])

                        rnn_inputs = tf.transpose(rnn_inputs, (1, 0, 2))

                        # create rnn
                        rnn_outputs, _, _ = cudnn_rnn(
                            input_data=rnn_inputs,
                            input_h=istate_h,
                            input_c=istate_c,
                            params=cudnn_rnn_params)

                        # import IPython; IPython.embed()

                        # ### initialize cudnn
                        # cudnn_rnn = CudnnLSTM(num_layers=self._N,
                        #                       num_units=self._rnn_state_dim,
                        #                       input_size=self._rnn_state_dim)
                        #
                        # ### istate_h
                        # istate_h_gt0 = tf.get_variable(name='istate_h_gt0',
                        #                                initializer=tf.random_uniform([self._batch_size,
                        #                                                               self._N - 1,
                        #                                                               self._rnn_state_dim]),
                        #                                validate_shape=False)
                        # ### istate_c
                        # istate_c_gt0 = tf.get_variable(name='istate_c_gt0',
                        #                                initializer=tf.random_uniform([self._batch_size,
                        #                                                               self._N - 1,
                        #                                                               self._rnn_state_dim]),
                        #                                validate_shape=False)
                        # ### cudnn params
                        # cudnn_rnn_params = tf.get_variable(name='cudnn_rnn_params',
                        #                                    # shape=cudnn_rnn.params_size(),
                        #                                    initializer=tf.random_uniform([cudnn_rnn.params_size()]),
                        #                                    validate_shape=False)
                        # # pad
                        # dyn_batch_size = tf.shape(rnn_inputs)[0]
                        # num_pa



                    else:
                        rnn_cell = tf.nn.rnn_cell.BasicRNNCell(self._rnn_state_dim, activation=self._rnn_activation)
                        rnn_outputs, rnn_states = tf.nn.dynamic_rnn(rnn_cell, rnn_inputs, initial_state=istate)
                    # rnn_outputs, rnn_states = tf.nn.dynamic_rnn(rnn_cell, rnn_inputs, initial_state=istate)

            # ### create rnn
            # with tf.name_scope('rnn'):
            #     with tf.variable_scope('rnn_vars'):
            #         if self._use_lstm:
            #             rnn_cell = tf.nn.rnn_cell.BasicLSTMCell(self._rnn_state_dim,
            #                                                     state_is_tuple=True,
            #                                                     activation=self._rnn_activation)
            #             istate = tf.nn.rnn_cell.LSTMStateTuple(*tf.split(1, 2, istate))  # so state_is_tuple=True
            #         else:
            #             rnn_cell = tf.nn.rnn_cell.BasicRNNCell(self._rnn_state_dim, activation=self._rnn_activation)
            #         rnn_outputs, rnn_states = tf.nn.dynamic_rnn(rnn_cell, rnn_inputs, initial_state=istate)

            ### final internal state --> reward
            with tf.name_scope('final_istate_to_reward'):
                # layer = rnn_outputs[:, -1, :]
                layer = rnn_outputs[-1, :, :]
                for i, num_outputs in enumerate(self._reward_hidden_layers + [1]):
                    activation = self._activation if i < len(self._reward_hidden_layers) else None
                    layer = layers.fully_connected(layer,
                                                   num_outputs=num_outputs,
                                                   activation_fn=activation,
                                                   weights_regularizer=layers.l2_regularizer(1.),
                                                   scope='rewards_i{0}'.format(i),
                                                   reuse=False)
                tf_rewards = layer

            tf_rewards = self._graph_preprocess_outputs(tf_rewards, d_preprocess)

        return tf_rewards


    def _graph_setup(self):
        tf_sess = tf.get_default_session()
        if tf_sess is None:
            tf_sess, tf_graph = Policy.create_session_and_graph(gpu_device=self._gpu_device, gpu_frac=self._gpu_frac)
        tf_graph = tf_sess.graph

        with tf_sess.as_default(), tf_graph.as_default():
            if ext.get_seed() is not None:
                ext.set_seed(ext.get_seed())

            ### policy
            with tf.variable_scope('policy'):
                d_preprocess = self._graph_preprocess_from_placeholders()
                tf_obs_ph, tf_actions_ph, tf_rewards_ph = self._graph_inputs_outputs_from_placeholders()
                tf_rewards = self._graph_inference(tf_obs_ph, tf_actions_ph, d_preprocess)
                tf_values = self._graph_calculate_values(tf_rewards)

            ### eval target network
            if self._separate_target_params:
                with tf.variable_scope('target_network'):
                    d_preprocess_target = self._graph_preprocess_from_placeholders()
                    tf_obs_target_ph, tf_actions_target_ph, _ = self._graph_inputs_outputs_from_placeholders()
                    tf_target_mask_ph = tf.placeholder('float', [None], name='tf_target_mask_ph')
                tf_target_rewards_eval_list = []
                for i, tf_obs_target_ph_i in enumerate(tf.split(0, self._batch_size, tf_obs_target_ph)):
                    with tf.variable_scope('target_network', reuse=(i > 0)):
                        tf_target_rewards_eval_list.append(self._graph_inference(tf_obs_target_ph_i,
                                                                                 tf_actions_target_ph,
                                                                                 d_preprocess_target))
                # print('eval target network'); import IPython; IPython.embed()
                tf_target_rewards_eval = tf.concat(0, tf_target_rewards_eval_list)
            else:
                with tf.variable_scope('target_network'):
                    tf_obs_target_ph, tf_actions_target_ph, _ = self._graph_inputs_outputs_from_placeholders()
                    tf_target_mask_ph = tf.placeholder('float', [None], name='tf_target_mask_ph')
                tf_target_rewards_eval_list = []
                for tf_obs_target_ph_i in tf.split(0, self._batch_size, tf_obs_target_ph):
                    with tf.variable_scope('policy', reuse=True):
                        tf_target_rewards_eval_list.append(self._graph_inference(tf_obs_target_ph_i,
                                                                                 tf_actions_target_ph,
                                                                                 d_preprocess))
                tf_target_rewards_eval = tf.concat(0, tf_target_rewards_eval_list)
            ### selection target network
            tf_target_rewards_select_list = []
            for i, tf_obs_target_ph_i in enumerate(tf.split(0, self._batch_size, tf_obs_target_ph)):
                with tf.variable_scope('policy', reuse=True):
                    tf_target_rewards_select_list.append(self._graph_inference(tf_obs_target_ph_i,
                                                                               tf_actions_target_ph,
                                                                               d_preprocess))
                tf_target_rewards_select = tf.concat(0, tf_target_rewards_select_list)

            policy_vars = sorted(tf.get_collection(xplatform.global_variables_collection_name(),
                                                   scope='policy'), key=lambda v: v.name)
            if self._separate_target_params:
                target_network_vars = sorted(tf.get_collection(xplatform.global_variables_collection_name(),
                                                               scope='target_network'), key=lambda v: v.name)
                update_target_fn = []
                for var, var_target in zip(policy_vars, target_network_vars):
                    assert (var.name.replace('policy', '') == var_target.name.replace('target_network', ''))
                    update_target_fn.append(var_target.assign(var))
                update_target_fn = tf.group(*update_target_fn)
            else:
                update_target_fn = None

            ### optimization
            tf_cost, tf_mse = self._graph_cost(tf_rewards_ph, tf_actions_ph, tf_rewards,
                                               tf_target_rewards_select, tf_target_rewards_eval, tf_target_mask_ph)
            tf_opt, tf_lr_ph = self._graph_optimize(tf_cost, policy_vars)

            self._graph_init_vars(tf_sess)

            # merged = tf.merge_all_summaries()
            # writer = tf.train.SummaryWriter('/tmp', graph_def=tf_sess.graph_def)

        return tf_graph, tf_sess, d_preprocess, \
               tf_obs_ph, tf_actions_ph, tf_rewards_ph, tf_values, tf_cost, tf_mse, tf_opt, tf_lr_ph, \
               tf_obs_target_ph, tf_actions_target_ph, tf_target_mask_ph, update_target_fn


    @overrides
    def train_step(self, step, observations, actions, rewards, dones, use_target):
        batch_size = len(observations)
        action_dim = self._env_spec.action_space.flat_dim

        policy_observations = observations[:, 0, :]
        policy_actions = actions[:, :self._H, :].reshape((batch_size, self._H * action_dim))
        policy_rewards = rewards[:, :self._N]
        target_observations = observations[:, self._N, :]

        if self._get_target_action_params['type'] == 'random':
            target_actions = self._get_random_action(self._get_target_action_params['K'])
        elif self._get_target_action_params['type'] == 'lattice':
            target_actions = self._get_target_action_preprocess['actions']
        else:
            raise NotImplementedError('get_action type {0} not implemented'.format(self._get_target_action_params['type']))

        feed_dict = {
            ### parameters
            self._tf_lr_ph: self._lr_schedule.value(step),
            ### policy
            self._tf_obs_ph: policy_observations,
            self._tf_actions_ph: policy_actions,
            self._tf_rewards_ph: policy_rewards,
            ### target network
            self._tf_obs_target_ph: target_observations,
            self._tf_actions_target_ph: target_actions,
            self._tf_target_mask_ph: float(use_target) * (1 - dones[:, self._N].astype(float))
        }
        # print('train_step'); import IPython; IPython.embed()
        start = time.time()
        cost, mse, _ = self._tf_sess.run([self._tf_cost, self._tf_mse, self._tf_opt], feed_dict=feed_dict)
        elapsed = time.time() - start

        assert (np.isfinite(cost))

        self._log_stats['Cost'].append(cost)
        self._log_stats['mse/cost'].append(mse / cost)
        self._log_stats['sess_run_time'].append(elapsed)