import numpy as np
import tensorflow as tf

from rllab.core.serializable import Serializable
from rllab.misc import ext

from sandbox.gkahn.rnn_critic.policies.mac_policy import MACPolicy
from sandbox.gkahn.rnn_critic.utils import tf_utils
from sandbox.gkahn.tf.core import xplatform
from sandbox.gkahn.rnn_critic.tf import networks

from sandbox.rocky.tf.spaces.discrete import Discrete


class RCcarMACPolicy(MACPolicy, Serializable):
    def __init__(self, **kwargs):
        Serializable.quick_init(self, locals())

        self._speed_weight = kwargs['speed_weight']
        self._is_classification = kwargs['is_classification']
        self._probcoll_strictly_increasing = kwargs['probcoll_strictly_increasing']
        self._coll_weight_pct = kwargs['coll_weight_pct']

        MACPolicy.__init__(self, **kwargs)

        assert(self._H == self._N)

    ###########################
    ### TF graph operations ###
    ###########################

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
        N = self._N
        assert(self._H == self._N)
        # tf.assert_equal(tf.shape(tf_obs_lowd)[0], tf.shape(tf_actions_ph)[0])

        self._action_graph.update({'output_dim': self._observation_graph['output_dim']})
        action_dim = tf_actions_ph.get_shape()[2].value
        actions = tf.reshape(tf_actions_ph, (-1, action_dim))
        rnn_inputs, _ = networks.fcnn(actions, self._action_graph, is_training=is_training, scope='fcnn_actions', T=H)
        rnn_inputs = tf.reshape(rnn_inputs, (-1, H, self._action_graph['output_dim']))

        rnn_outputs, _ = networks.rnn(rnn_inputs, self._rnn_graph, initial_state=tf_obs_lowd)
        rnn_output_dim = rnn_outputs.get_shape()[2].value
        rnn_outputs = tf.reshape(rnn_outputs, (-1, rnn_output_dim))

        self._output_graph.update({'output_dim': 1})
        tf_values, _ = networks.fcnn(rnn_outputs, self._output_graph, is_training=is_training, scope='fcnn_values', T=H)
        tf_values = tf.reshape(tf_values, (-1, H))

        if self._probcoll_strictly_increasing:
            tf_values = tf_utils.cumulative_increasing_sum(tf_values)

        if values_softmax['type'] == 'final':
            tf_values_softmax = tf.zeros([batch_size, N])
            tf_values_softmax[:, -1] = 1.
        elif values_softmax['type'] == 'mean':
            tf_values_softmax = (1. / float(N)) * tf.ones([batch_size, N])
        elif values_softmax['type'] == 'exponential':
            lam = values_softmax['exponential']['lambda']
            lams = (1 - lam) * np.power(lam, np.arange(N - 1))
            lams = np.array(list(lams) + [np.power(lam, N - 1)])
            tf_values_softmax = lams * tf.ones(tf.shape(tf_values))
        else:
            raise NotImplementedError

        assert(tf_values.get_shape()[1].value == H)

        return tf_values, tf_values_softmax, None, None

    def _graph_get_action(self, tf_obs_ph, get_action_params, scope_select, reuse_select, scope_eval, reuse_eval,
                          add_speed_cost=False):
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
        max_speed = self._env_spec.action_space.high[1]

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
        else:
            raise NotImplementedError

        ### process to lowd
        with tf.variable_scope(scope_select, reuse=reuse_select):
            tf_preprocess_select = self._graph_preprocess_placeholders()
            tf_obs_lowd_select = self._graph_obs_to_lowd(tf_obs_ph, tf_preprocess_select, is_training=False)
        with tf.variable_scope(scope_eval, reuse=reuse_eval):
            tf_preprocess_eval = self._graph_preprocess_placeholders()
            tf_obs_lowd_eval = self._graph_obs_to_lowd(tf_obs_ph, tf_preprocess_eval, is_training=False)
        ### tile
        tf_actions = tf.tile(tf_actions, (num_obs, 1, 1))
        tf_obs_lowd_repeat_select = tf_utils.repeat_2d(tf_obs_lowd_select, K, 0)
        tf_obs_lowd_repeat_eval = tf_utils.repeat_2d(tf_obs_lowd_eval, K, 0)
        ### inference to get values
        with tf.variable_scope(scope_select, reuse=reuse_select):
            tf_values_all_select, tf_values_softmax_all_select, _, _ = \
                self._graph_inference(tf_obs_lowd_repeat_select, tf_actions, get_action_params['values_softmax'],
                                      tf_preprocess_select, is_training=False, add_reg=False)  # [num_obs*k, H]
        with tf.variable_scope(scope_eval, reuse=reuse_eval):
            tf_values_all_eval, tf_values_softmax_all_eval, _, _ = \
                self._graph_inference(tf_obs_lowd_repeat_eval, tf_actions, get_action_params['values_softmax'],
                                      tf_preprocess_eval, is_training=False, add_reg=False)  # [num_obs*k, H]
        if self._is_classification:
            ### convert pre-activation to post-activation
            tf_values_all_select = -tf.sigmoid(tf_values_all_select)
            tf_values_all_eval = -tf.sigmoid(tf_values_all_eval)
        else:
            tf_values_all_select = -tf_values_all_select
            tf_values_all_eval = -tf_values_all_eval
        ### get_action based on select (policy)
        tf_values_select = tf.reduce_sum(tf_values_all_select * tf_values_softmax_all_select, reduction_indices=1)  # [num_obs*K]
        if add_speed_cost:
            tf_values_select -= self._speed_weight * tf.reduce_mean(tf.square(tf_actions[:, :, 1] - max_speed),
                                                                    reduction_indices=1)
        tf_values_select = tf.reshape(tf_values_select, (num_obs, K))  # [num_obs, K]
        tf_values_argmax_select = tf.one_hot(tf.argmax(tf_values_select, 1), depth=K)  # [num_obs, K]
        tf_get_action = tf.reduce_sum(
            tf.tile(tf.expand_dims(tf_values_argmax_select, 2), (1, 1, action_dim)) *
            tf.reshape(tf_actions, (num_obs, K, H, action_dim))[:, :, 0, :],
            reduction_indices=1)  # [num_obs, action_dim]
        ### get_action_value based on eval (target)
        tf_values_eval = tf.reduce_sum(tf_values_all_eval * tf_values_softmax_all_eval, reduction_indices=1)  # [num_obs*K]
        if add_speed_cost:
            tf_values_eval -= self._speed_weight * tf.reduce_mean(tf.square(tf_actions[:, :, 1] - max_speed),
                                                                    reduction_indices=1)
        tf_values_eval = tf.reshape(tf_values_eval, (num_obs, K))  # [num_obs, K]
        tf_get_action_value = tf.reduce_sum(tf_values_argmax_select * tf_values_eval, reduction_indices=1)

        ### check shapes
        tf.assert_equal(tf.shape(tf_get_action)[0], num_obs)
        tf.assert_equal(tf.shape(tf_get_action_value)[0], num_obs)
        assert(tf_get_action.get_shape()[1].value == action_dim)

        return tf_get_action, tf_get_action_value

    def _graph_cost(self, tf_train_values, tf_train_values_softmax, tf_rewards_ph, tf_dones_ph,
                    tf_target_get_action_values):
        if not self._is_classification:
            return MACPolicy._graph_cost(self, tf_train_values, tf_train_values_softmax, tf_rewards_ph, tf_dones_ph,
                                         tf_target_get_action_values)

        tf_dones = tf.cast(tf_dones_ph, tf.int32)
        tf_labels = tf.cast(tf.cumsum(tf_rewards_ph, axis=1) < -0.5, tf.float32)

        ### mask
        if self._clip_cost_target_with_dones:
            lengths = tf.reduce_sum(1 - tf_dones, axis=1) + 1
            all_mask = tf.sequence_mask(
                tf.cast(lengths, tf.int32),
                maxlen=self._H,
                dtype=tf.float32)
            if self._coll_weight_pct is not None:
                factor = tf.reduce_sum(all_mask * tf_labels) / tf.reduce_sum(tf.cast(tf.reduce_sum(lengths), tf.float32))
                coll_weight = (self._coll_weight_pct - self._coll_weight_pct * factor) / (factor - self._coll_weight_pct * factor)
                one_mask = tf.one_hot(
                    tf.cast(lengths - 1, tf.int32),
                    self._H) * (coll_weight - 1.)
                one_mask_coll = one_mask * tf_labels
                mask = tf.cast(all_mask + one_mask_coll, tf.float32)
            else:
                mask = all_mask
        else:
            mask = tf.ones(tf.shape(tf_labels), dtype=tf.float32)
        mask /= tf.reduce_sum(mask)

        ### desired values
        if self._use_target:
            control_dependencies = []
            control_dependencies += [tf.assert_greater_equal(tf_target_get_action_values, -1., name='cost_assert_0')]
            control_dependencies += [tf.assert_less_equal(tf_target_get_action_values, 0., name='cost_assert_1')]
            with tf.control_dependencies(control_dependencies):
                tf_labels = tf.clip_by_value(tf_labels +
                                             (1 - tf.cast(tf_dones, tf.float32)) * -tf.round(tf_target_get_action_values),
                                             0., 1.)

        ### cost
        control_dependencies = []
        control_dependencies += [tf.assert_greater_equal(tf_labels, 0., name='cost_assert_2')]
        control_dependencies += [tf.assert_less_equal(tf_labels, 1., name='cost_assert_3')]
        with tf.control_dependencies(control_dependencies):
            cross_entropies = tf.nn.sigmoid_cross_entropy_with_logits(logits=tf_train_values, labels=tf_labels)
            cost = tf.reduce_sum(mask * cross_entropies)
            weight_decay = self._weight_decay * tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))

        return cost + weight_decay, cost

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
            tf_obs_ph, tf_actions_ph, tf_dones_ph, tf_rewards_ph, tf_obs_target_ph, \
            tf_test_es_ph_dict = self._graph_input_output_placeholders()

            ### policy
            policy_scope = 'policy'
            with tf.variable_scope(policy_scope):
                ### create preprocess placeholders
                tf_preprocess = self._graph_preprocess_placeholders()
                ### process obs to lowd
                tf_obs_lowd = self._graph_obs_to_lowd(tf_obs_ph, tf_preprocess, is_training=True)
                ### create training policy
                tf_train_values, tf_train_values_softmax, _, _ = \
                    self._graph_inference(tf_obs_lowd, tf_actions_ph[:, :self._H, :],
                                          self._values_softmax, tf_preprocess, is_training=True)

            with tf.variable_scope(policy_scope, reuse=True):
                tf_train_values_test, tf_train_values_softmax_test, _, _ = \
                    self._graph_inference(tf_obs_lowd, tf_actions_ph[:, :self._get_action_test['H'], :],
                                          self._values_softmax, tf_preprocess, is_training=False)
                tf_get_value = tf.reduce_sum(tf_train_values_softmax_test * tf_train_values_test, reduction_indices=1)

            ### action selection
            tf_get_action, tf_get_action_value = self._graph_get_action(tf_obs_ph, self._get_action_test,
                                                                        policy_scope, True, policy_scope, True,
                                                                        add_speed_cost=True)
            ### exploration strategy and logprob
            tf_get_action_explore = self._graph_get_action_explore(tf_get_action, tf_test_es_ph_dict)

            ### get policy variables
            tf_policy_vars = sorted(tf.get_collection(xplatform.global_variables_collection_name(),
                                                      scope=policy_scope), key=lambda v: v.name)
            tf_trainable_policy_vars = sorted(tf.get_collection(xplatform.trainable_variables_collection_name(),
                                                                scope=policy_scope), key=lambda v: v.name)

            ### create target network
            if self._use_target:
                target_scope = 'target' if self._separate_target_params else 'policy'
                ### action selection
                tf_obs_target_ph_packed = xplatform.concat([tf_obs_target_ph[:, h - self._obs_history_len:h, :]
                                                     for h in range(self._obs_history_len, self._obs_history_len + self._N + 1)],
                                                    0)
                tf_target_get_action, tf_target_get_action_values = self._graph_get_action(tf_obs_target_ph_packed,
                                                                                           self._get_action_target,
                                                                                           scope_select=policy_scope,
                                                                                           reuse_select=True,
                                                                                           scope_eval=target_scope,
                                                                                           reuse_eval=(target_scope == policy_scope))

                tf_target_get_action_values = tf.transpose(tf.reshape(tf_target_get_action_values, (self._N + 1, -1)))[:, 1:]
            else:
                tf_target_get_action_values = tf.zeros([tf.shape(tf_train_values)[0], self._N])

            ### update target network
            if self._use_target and self._separate_target_params:
                tf_policy_vars_nobatchnorm = list(
                    filter(lambda v: 'biased' not in v.name and 'local_step' not in v.name,
                           tf_policy_vars))
                tf_target_vars = sorted(tf.get_collection(xplatform.global_variables_collection_name(),
                                                          scope=target_scope), key=lambda v: v.name)
                assert (len(tf_policy_vars_nobatchnorm) == len(tf_target_vars))
                tf_update_target_fn = []
                for var, var_target in zip(tf_policy_vars_nobatchnorm, tf_target_vars):
                    assert (var.name.replace(policy_scope, '') == var_target.name.replace(target_scope, ''))
                    tf_update_target_fn.append(var_target.assign(var))
                tf_update_target_fn = tf.group(*tf_update_target_fn)
            else:
                tf_update_target_fn = None

            ### optimization
            tf_cost, tf_mse = self._graph_cost(tf_train_values, tf_train_values_softmax, tf_rewards_ph, tf_dones_ph,
                                               tf_target_get_action_values)
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
            'test_es_ph_dict': tf_test_es_ph_dict,
            'preprocess': tf_preprocess,
            'get_value': tf_get_value,
            'get_action': tf_get_action,
            'get_action_explore': tf_get_action_explore,
            'get_action_value': tf_get_action_value,
            'update_target_fn': tf_update_target_fn,
            'cost': tf_cost,
            'mse': tf_mse,
            'opt': tf_opt,
            'lr_ph': tf_lr_ph,
        }

    ################
    ### Training ###
    ################

    def train_step(self, step, steps, observations, actions, rewards, values, dones, logprobs, use_target):
        # always True use_target so dones is passed in
        return MACPolicy.train_step(self, step, steps, observations, actions, rewards, values, dones, logprobs,
                                    self._is_classification or use_target)

