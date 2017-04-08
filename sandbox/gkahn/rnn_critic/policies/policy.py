import abc
import os
from collections import defaultdict
import itertools
import time

import numpy as np
import scipy.linalg
import tensorflow as tf
from sklearn.utils.extmath import cartesian

from rllab.core.serializable import Serializable
import rllab.misc.logger as logger
from rllab.misc import ext

from sandbox.rocky.tf.spaces.discrete import Discrete
from sandbox.gkahn.tf.policies.base import Policy as TfPolicy
from sandbox.gkahn.tf.core import xplatform
from sandbox.gkahn.rnn_critic.utils import schedules, tf_utils

class Policy(TfPolicy, Serializable):
    __metaclass__ = abc.ABCMeta

    def __init__(self,
                 env_spec,
                 N,
                 H,
                 cost_type,
                 gamma,
                 obs_history_len,
                 use_target,
                 separate_target_params,
                 weight_decay,
                 lr_schedule,
                 grad_clip_norm,
                 get_action_params,
                 get_target_action_params,
                 preprocess,
                 gpu_device=None,
                 gpu_frac=None,
                 **kwargs):
        """
        :param N: number of returns to use (i.e. n-step)
        :param H: action planning horizon
        :param cost_type: combined or separated
        :param gamma: reward decay
        :param obs_history_len: how many previous obs to include when sampling? (=1 is only current observation)
        :param use_target: include target network or not
        :param separate_target_params: if True, have separate params that you copy over periodically
        :param weight_decay
        :param lr_schedule: arguments for PiecewiseSchedule
        :param grad_clip_norm
        :param get_action_params: how should actions be chosen?
        :param get_target_action_params: how should target actions be chosen?
        :param reset_every_train: reset parameters every time train is called?
        :param train_steps: how many calls to optimizer each time train is called
        :param batch_size
        :param get_action_params: dictionary specifying how to choose actions
        """
        Serializable.quick_init(self, locals())
        assert(cost_type == 'combined' or cost_type == 'separated')
        assert(type(obs_history_len) is int and obs_history_len >= 1)

        self._env_spec = env_spec
        self._N = N
        self._H = H
        self._cost_type = cost_type
        self._gamma = gamma
        self._obs_history_len = obs_history_len
        self._use_target = use_target
        self._separate_target_params = separate_target_params
        self._weight_decay = weight_decay
        self._lr_schedule = schedules.PiecewiseSchedule(**lr_schedule)
        self._grad_clip_norm = grad_clip_norm
        self._preprocess_params = preprocess
        self._get_action_params = get_action_params
        self._get_target_action_params = get_target_action_params
        self._gpu_device = gpu_device
        self._gpu_frac = gpu_frac
        self._exploration_strategy = None # don't set in init b/c will then be Serialized

        self._tf_debug = dict()
        self._tf_graph, self._tf_sess, self._d_preprocess, \
            self._tf_obs_ph, self._tf_actions_ph, self._tf_rewards_ph, self._tf_values, self._tf_cost, self._tf_mse, self._tf_opt, self._tf_lr_ph, \
            self._tf_obs_target_ph, self._tf_actions_target_ph, self._tf_target_mask_ph, self._update_target_fn = \
            self._graph_setup()

        self._get_action_preprocess = self._get_action_setup(self._get_action_params)
        self._get_target_action_preprocess = self._get_action_setup(self._get_target_action_params)
        self._num_get_action = 0 # keep track of how many times get action called

        ### saving/loading
        self._match_dict = dict()

        ### logging
        self._log_stats = defaultdict(list)

        TfPolicy.__init__(self, env_spec, sess=self._tf_sess)

    ##################
    ### Properties ###
    ##################

    @property
    def N(self):
        return self._N

    @abc.abstractproperty
    def N_output(self):
        """ number of outputs of graph inference """
        raise NotImplementedError

    @property
    def session(self):
        return self._tf_sess

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

        tf_graph = tf.Graph()
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_device)
        if len(str(gpu_device)) > 0:
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_frac)
            config = tf.ConfigProto(gpu_options=gpu_options,
                                    log_device_placement=False,
                                    allow_soft_placement=True)
        else:
            config = tf.ConfigProto(
                device_count={'GPU': 0}
            )
        tf_sess = tf.Session(graph=tf_graph, config=config)
        return tf_sess, tf_graph

    def _graph_inputs_outputs_from_placeholders(self):
        with tf.variable_scope('feed_input'):
            obs_shape = self._env_spec.observation_space.shape
            tf_obs_ph = tf.placeholder(tf.uint8 if len(obs_shape) > 1 else tf.float32,
                                       [None, self._env_spec.observation_space.flat_dim * self._obs_history_len], name='tf_obs_ph')
            tf_actions_ph = tf.placeholder(tf.float32, [None, self._env_spec.action_space.flat_dim * self._H], name='tf_actions_ph')
            tf_rewards_ph = tf.placeholder(tf.float32, [None, self._N], name='tf_rewards_ph')

        return tf_obs_ph, tf_actions_ph, tf_rewards_ph

    def _graph_preprocess_from_placeholders(self):
        d_preprocess = dict()

        with tf.variable_scope('preprocess'):
            for name, dim, diag_orth in (('observations', self._env_spec.observation_space.flat_dim, self._obs_is_im),
                                         ('actions', self._env_spec.action_space.flat_dim * self._H, False),
                                         ('rewards', self.N_output, False)):
                d_preprocess[name+'_mean_ph'] = tf.placeholder(tf.float32, shape=(1, dim), name=name+'_mean_ph')
                d_preprocess[name+'_mean_var'] = tf.get_variable(name+'_mean_var', shape=[1, dim],
                                                                 trainable=False, dtype=tf.float32,
                                                                 initializer=tf.constant_initializer(np.zeros((1, dim))))
                d_preprocess[name+'_mean_assign'] = tf.assign(d_preprocess[name+'_mean_var'],
                                                              d_preprocess[name+'_mean_ph'])

                if diag_orth:
                    d_preprocess[name + '_orth_ph'] = tf.placeholder(tf.float32, shape=(dim,), name=name + '_orth_ph')
                    d_preprocess[name + '_orth_var'] = tf.get_variable(name + '_orth_var',
                                                                       shape=(dim,),
                                                                       trainable=False, dtype=tf.float32,
                                                                       initializer=tf.constant_initializer(np.ones(dim)))
                else:
                    d_preprocess[name + '_orth_ph'] = tf.placeholder(tf.float32, shape=(dim, dim), name=name + '_orth_ph')
                    d_preprocess[name + '_orth_var'] = tf.get_variable(name + '_orth_var',
                                                                       shape=(dim, dim),
                                                                       trainable=False, dtype=tf.float32,
                                                                       initializer=tf.constant_initializer(np.eye(dim)))
                d_preprocess[name+'_orth_assign'] = tf.assign(d_preprocess[name+'_orth_var'],
                                                              d_preprocess[name+'_orth_ph'])

        return d_preprocess

    def _graph_preprocess_inputs(self, tf_obs_ph, tf_actions_ph, d_preprocess):
        ### whiten inputs
        if tf_obs_ph.dtype != tf.float32:
            tf_obs_ph = tf.cast(tf_obs_ph, tf.float32)
        if self._obs_is_im:
            tf_obs_whitened = tf.mul(tf_obs_ph -
                                     tf.tile(d_preprocess['observations_mean_var'], (1, self._obs_history_len)),
                                     tf.tile(d_preprocess['observations_orth_var'], (self._obs_history_len,)))
        else:
            tf_obs_whitened = tf.matmul(tf_obs_ph -
                                        tf.tile(d_preprocess['observations_mean_var'], (1, self._obs_history_len)),
                                        tf_utils.block_diagonal([d_preprocess['observations_orth_var']] * self._obs_history_len))

        if isinstance(self._env_spec.action_space, Discrete):
            tf_actions_whitened = tf_actions_ph
        else:
            tf_actions_whitened = tf.matmul(tf_actions_ph - d_preprocess['actions_mean_var'],
                                            d_preprocess['actions_orth_var'])

        return tf_obs_whitened, tf_actions_whitened

    def _graph_match_actions(self, layer, tf_actions):
        num_layer = tf.shape(layer)[0]
        num_action = tf.shape(tf_actions)[0]

        # assumes num_action is a multiple of num_obs
        layer_cond = tf.cond(tf.not_equal(num_layer, num_action),
                             lambda: tf_utils.repeat_2d(layer, num_action // num_layer, 0),
                             lambda: layer)

        return layer_cond

    def _graph_preprocess_outputs(self, tf_rewards, d_preprocess):
        return tf.add(tf.matmul(tf_rewards, tf.transpose(d_preprocess['rewards_orth_var'])),
                      d_preprocess['rewards_mean_var'])

    @abc.abstractmethod
    def _graph_inference(self, tf_obs_ph, tf_actions_ph, d_preprocess):
        raise NotImplementedError

    def _graph_calculate_values(self, tf_rewards):
        N = tf_rewards.get_shape()[1].value
        gammas = np.power(self._gamma * np.ones(N), np.arange(N))
        tf_values = tf.reduce_sum(gammas * tf_rewards, reduction_indices=1)
        return tf_values

    def _graph_cost(self, tf_rewards_ph, tf_actions_ph, tf_rewards,
                    tf_target_rewards_select, tf_target_rewards_eval, tf_target_mask_ph):
        # for training, len(tf_obs_ph) == len(tf_actions_ph)
        # but len(tf_actions_target_ph) == N * len(tf_action_ph),
        # so need to be selective about what to take the max over

        ### calculate values
        tf_target_values_select = self._graph_calculate_values(tf_target_rewards_select)
        tf_target_values_eval = self._graph_calculate_values(tf_target_rewards_eval)
        ### flatten
        batch_size = tf.shape(tf_rewards_ph)[0]
        tf_target_values_select_flat = tf.reshape(tf_target_values_select, (batch_size, -1))
        tf_target_values_eval_flat = tf.reshape(tf_target_values_eval, (batch_size, -1))
        ### mask selection and eval
        tf_target_values_mask = tf.one_hot(tf.argmax(tf_target_values_select_flat, 1),
                                           depth=tf.shape(tf_target_values_select_flat)[1])
        tf_target_values_max = tf.reduce_sum(tf_target_values_mask * tf_target_values_eval_flat, reduction_indices=1)
        if self._use_target:
            tf_target_bootstrap = tf_target_mask_ph * np.power(self._gamma, self._N)*tf_target_values_max
        else:
            tf_target_bootstrap = 0

        tf_values_ph = self._graph_calculate_values(tf_rewards_ph)
        tf_values = self._graph_calculate_values(tf_rewards)

        mse = tf.reduce_mean(tf.square(tf_values_ph + tf_target_bootstrap - tf_values))
        if len(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)) > 0:
            weight_decay = self._weight_decay * tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
        else:
            weight_decay = 0
        cost = mse + weight_decay
        return cost, mse

    def _graph_optimize(self, tf_cost, policy_vars):
        tf_lr_ph = tf.placeholder(tf.float32, (), name="learning_rate")
        optimizer = tf.train.AdamOptimizer(learning_rate=tf_lr_ph, epsilon=1e-4)
        gradients = optimizer.compute_gradients(tf_cost, var_list=policy_vars)
        for i, (grad, var) in enumerate(gradients):
            if grad is not None:
                gradients[i] = (tf.clip_by_norm(grad, self._grad_clip_norm), var)
        return optimizer.apply_gradients(gradients), tf_lr_ph

    def _graph_init_vars(self, tf_sess):
        tf_sess.run([xplatform.global_variables_initializer()])

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
                    tf_target_rewards_eval = self._graph_inference(tf_obs_target_ph, tf_actions_target_ph, d_preprocess_target)
            else:
                with tf.variable_scope('target_network'):
                    tf_obs_target_ph, tf_actions_target_ph, _ = self._graph_inputs_outputs_from_placeholders()
                    tf_target_mask_ph = tf.placeholder('float', [None], name='tf_target_mask_ph')
                with tf.variable_scope('policy', reuse=True):
                    tf_target_rewards_eval = self._graph_inference(tf_obs_target_ph, tf_actions_target_ph, d_preprocess)
            ### selection target network
            with tf.variable_scope('policy', reuse=True):
                tf_target_rewards_select = self._graph_inference(tf_obs_target_ph, tf_actions_target_ph, d_preprocess)

            policy_vars = sorted(tf.get_collection(xplatform.global_variables_collection_name(),
                                                   scope='policy'), key=lambda v: v.name)
            if self._separate_target_params:
                target_network_vars = sorted(tf.get_collection(xplatform.global_variables_collection_name(),
                                                               scope='target_network'), key=lambda v: v.name)
                update_target_fn = []
                for var, var_target in zip(policy_vars, target_network_vars):
                    assert(var.name.replace('policy', '') == var_target.name.replace('target_network', ''))
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
                tf_assigns.append(self._d_preprocess[key + '_assign'])

        # we assume if obs is im, the obs orth is the diagonal of the covariance
        self._tf_sess.run(tf_assigns,
          feed_dict={
              self._d_preprocess['observations_mean_ph']: obs_mean,
              self._d_preprocess['observations_orth_ph']: obs_orth,
              self._d_preprocess['actions_mean_ph']: np.tile(actions_mean, self._H),
              self._d_preprocess['actions_orth_ph']: scipy.linalg.block_diag(*([actions_orth] * self._H)),
              self._d_preprocess['rewards_mean_ph']: (self._N / float(self.N_output)) * # reweighting!
                                                     np.expand_dims(np.tile(rewards_mean, self.N_output), 0),
              self._d_preprocess['rewards_orth_ph']: (self._N / float(self.N_output)) *
                                                     scipy.linalg.block_diag(*([rewards_orth] * self.N_output))
          })

    def update_target(self):
        if self._separate_target_params:
            self._tf_sess.run(self._update_target_fn)

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
        target_actions = np.tile(target_actions, (len(observations), 1))

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
        start = time.time()
        cost, mse, _ = self._tf_sess.run([self._tf_cost, self._tf_mse, self._tf_opt], feed_dict=feed_dict)
        elapsed = time.time() - start

        assert(np.isfinite(cost))

        self._log_stats['Cost'].append(cost)
        self._log_stats['mse/cost'].append(mse/cost)
        self._log_stats['sess_run_time'].append(elapsed)

    ######################
    ### Policy methods ###
    ######################

    def _get_action_setup(self, params):
        get_action_preprocess = dict()

        if params['type'] == 'random':
            pass
        elif params['type'] == 'lattice':
            if isinstance(self._env_spec.action_space, Discrete):
                action_dim = self._env_spec.action_space.n
                indices = cartesian([np.arange(action_dim)] * self._H) + np.r_[0:action_dim*self._H:action_dim]
                actions = np.zeros((len(indices), action_dim * self._H))
                for i, one_hots in enumerate(indices):
                    actions[i, one_hots] = 1
                get_action_preprocess['actions'] = actions
            else:
                K = params['K']
                action_lower, action_upper = self._env_spec.action_space.bounds
                single_actions = cartesian([np.linspace(l, u, K) for l, u in zip(action_lower, action_upper)])
                actions = np.asarray(list(itertools.combinations(single_actions, self._H)))
                get_action_preprocess['actions'] = actions.reshape((len(actions),
                                                                    self._H * self._env_spec.action_space.flat_dim))
        else:
            raise NotImplementedError('get_action type {0} not implemented'.format(params['type']))

        return get_action_preprocess

    def _get_random_action(self, N):
        u_dim = self._env_spec.action_space.flat_dim
        if isinstance(self._env_spec.action_space, Discrete):
            actions = np.random.randint(0, u_dim, size=(N, self._H))
            actions = (np.arange(u_dim) == actions[:, :, None]).astype(int)
            actions = actions.reshape(N, self._H * u_dim).astype(float)
        else:
            action_lower, action_upper = self._env_spec.action_space.bounds
            actions = np.random.uniform(action_lower.tolist(), action_upper.tolist(),
                                        size=(N, self._H, u_dim))
            actions = actions.reshape(N, self._H * u_dim)

        return actions

    def set_exploration_strategy(self, exploration_strategy):
        self._exploration_strategy = exploration_strategy

    def get_action(self, observation, return_action_info=False):
        chosen_actions, action_info = self.get_actions([observation], return_action_info=return_action_info)
        return chosen_actions[0], action_info

    def get_actions(self, observations, return_action_info=False):
        num_obs = len(observations)
        observations = self._env_spec.observation_space.flatten_n(observations)

        if self._get_action_params['type'] == 'random':
            actions = self._get_random_action(self._get_action_params['K'] * num_obs)
        elif self._get_action_params['type'] == 'lattice':
            actions = np.tile(self._get_action_preprocess['actions'], (num_obs, 1))
        else:
            raise NotImplementedError('get_actions type {0} not implemented'.format(self._get_action_params['type']))

        pred_values = self._tf_sess.run([self._tf_values],
                                        feed_dict={self._tf_obs_ph: observations,
                                                   self._tf_actions_ph: actions})[0]

        chosen_actions = []
        for i, (observation_i, pred_values_i, actions_i) in enumerate(zip(observations,
                                                                          np.split(pred_values, num_obs, axis=0),
                                                                          np.split(actions, num_obs, axis=0))):
            chosen_action_i = actions_i[pred_values_i.argmax()][:self._env_spec.action_space.flat_dim]
            if isinstance(self._env_spec.action_space, Discrete):
                chosen_action_i = int(chosen_action_i.argmax())
            if self._exploration_strategy is not None:
                exploration_func = lambda: None
                exploration_func.get_action = lambda _: (chosen_action_i, dict())
                chosen_action_i = self._exploration_strategy.get_action(self._num_get_action + i,
                                                                        observation_i,
                                                                        exploration_func)
            chosen_actions.append(chosen_action_i)

        self._num_get_action += num_obs

        if return_action_info:
            action_info = {
                'actions': np.split(actions, num_obs, axis=0),
                'values': np.split(pred_values, num_obs, axis=0)
            }
        else:
            action_info = dict()

        return chosen_actions, action_info

    @property
    def recurrent(self):
        return False

    def terminate(self):
        self._tf_sess.close()

    ##################
    ### Evaluation ###
    ##################

    def eval_Q_values(self, observations, actions):
        Q_values = self._tf_sess.run(self._tf_values,
                                     feed_dict={self._tf_obs_ph: observations,
                                                self._tf_actions_ph: actions})
        return Q_values

    ######################
    ### Saving/loading ###
    ######################

    def get_params_internal(self, **tags):
        with self._tf_graph.as_default():
            return sorted(tf.get_collection(xplatform.global_variables_collection_name()), key=lambda v: v.name)

    ###############
    ### Logging ###
    ###############

    def log(self):
        for k in sorted(self._log_stats.keys()):
            logger.record_tabular(k, np.mean(self._log_stats[k]))
        self._log_stats.clear()
