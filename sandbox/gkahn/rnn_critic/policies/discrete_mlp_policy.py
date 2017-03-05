import numpy as np
import tensorflow as tf

from sandbox.rocky.tf.spaces.discrete import Discrete
from rllab.core.serializable import Serializable
from sandbox.gkahn.rnn_critic.policies.policy import RNNCriticPolicy

class RNNCriticDiscreteMLPPolicy(RNNCriticPolicy, Serializable):
    def __init__(self,
                 hidden_layers,
                 activation,
                 **kwargs):
        """
        :param hidden_layers: list of layer sizes
        :param activation: str to be evaluated (e.g. 'tf.nn.relu')
        """
        assert(isinstance(kwargs.get('env_spec').action_space, Discrete))

        Serializable.quick_init(self, locals())

        self._hidden_layers = list(hidden_layers)
        self._activation = eval(activation)

        RNNCriticPolicy.__init__(self, **kwargs)

    ###########################
    ### TF graph operations ###
    ###########################

    def _graph_preprocess_from_placeholders(self):
        d_preprocess = dict()

        with tf.variable_scope('preprocess'):
            for name, dim in (('observations', self._env_spec.observation_space.flat_dim),
                              ('rewards', 1)):
                d_preprocess[name+'_mean_ph'] = tf.placeholder(tf.float32, shape=(1, dim), name=name+'_mean_ph')
                d_preprocess[name+'_mean_var'] = tf.get_variable(name+'_mean_var', shape=[1, dim],
                                                                 trainable=False, dtype=tf.float32,
                                                                 initializer=tf.constant_initializer(np.zeros((1, dim))))
                d_preprocess[name+'_mean_assign'] = tf.assign(d_preprocess[name+'_mean_var'],
                                                              d_preprocess[name+'_mean_ph'])

                d_preprocess[name+'_orth_ph'] = tf.placeholder(tf.float32, shape=(dim, dim), name=name+'_orth_ph')
                d_preprocess[name+'_orth_var'] = tf.get_variable(name+'_orth_var',
                                                                 shape=(dim, dim),
                                                                 trainable=False, dtype=tf.float32,
                                                                 initializer=tf.constant_initializer(np.eye(dim)))
                d_preprocess[name+'_orth_assign'] = tf.assign(d_preprocess[name+'_orth_var'],
                                                              d_preprocess[name+'_orth_ph'])

        return d_preprocess

    def _graph_preprocess_inputs(self, tf_obs_ph, d_preprocess):
        ### whiten inputs
        tf_obs_whitened = tf.matmul(tf_obs_ph - d_preprocess['observations_mean_var'],
                                    d_preprocess['observations_orth_var'])

        return tf_obs_whitened

    def _graph_preprocess_outputs(self, tf_rewards, d_preprocess):
        return (tf_rewards * d_preprocess['rewards_orth_var'][0,0]) + d_preprocess['rewards_mean_var']

    def _graph_inference(self, tf_obs_ph, tf_actions_ph, d_preprocess):
        input_shape = self._env_spec.observation_space.shape
        action_dim = self._env_spec.action_space.flat_dim
        assert(len(input_shape) == 1) # TODO

        input_dim = np.prod(input_shape)
        output_dim = np.power(action_dim, self._H)

        with tf.name_scope('inference'):

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
                tf.add_to_collection('weight_decays', 0.5 * tf.reduce_mean(tf.square(v)))

            ### fully connected
            tf_obs = self._graph_preprocess_inputs(tf_obs_ph, d_preprocess)
            layer = tf_obs
            for i, (weight, bias) in enumerate(zip(weights, biases)):
                with tf.name_scope('hidden_{0}'.format(i)):
                    layer = tf.add(tf.matmul(layer, weight), bias)
                    if i < len(weights) - 1:
                        layer = self._activation(layer)
            rewards_all = layer

            tf_rewards = self._graph_preprocess_outputs(rewards_all, d_preprocess)

        return tf_rewards

    def _graph_calculate_values(self, tf_rewards):
        return tf_rewards # b/c network outputs values directly

    def _graph_cost(self, tf_rewards_ph, tf_actions_ph, tf_rewards, tf_target_rewards, tf_target_mask_ph):
        with tf.name_scope('cost'):
            ### values of label rewards
            gammas = np.power(self._gamma * np.ones(self._H), np.arange(self._H))
            tf_values_ph = tf.reduce_sum(gammas * tf_rewards_ph, reduction_indices=1)
            ## target network max
            tf_target_values_max = tf.reduce_max(tf_target_rewards, reduction_indices=1)

            ### values of policy
            action_dim = self._env_spec.action_space.flat_dim
            with tf.name_scope('reward_idxs'):
                # action_dim = 4
                # H = 2
                # tf_actions_ph
                # [0, 0, 0, 1, 0, 0, 0, 1]
                tf_b = tf.tile(tf.expand_dims(tf.range(action_dim), 0), (1, self._H)) * tf.cast(tf_actions_ph, tf.int32)
                # [0, 0, 0, 3, 0, 0, 0, 3]
                tf_c = tf_b * np.power(action_dim, tf.constant(np.repeat(np.arange(self._H), (action_dim,)), dtype=tf.int32))
                # [0, 0, 0, 3, 0, 0, 0, 12]
                reward_idxs = tf.reduce_sum(tf_c, axis=1)
                # [15]

            ### extract relevant sum of rewards
            with tf.name_scope('gather_rewards'):
                reward_idxs_flat = reward_idxs + tf.range(tf.shape(reward_idxs)[0]) * np.power(action_dim, self._H)
                tf_values = tf.gather(tf.reshape(tf_rewards, [-1]), reward_idxs_flat)

            self._tf_debug['tf_actions_ph'] = tf_actions_ph
            self._tf_debug['tf_b'] = tf_b
            self._tf_debug['tf_c'] = tf_c
            self._tf_debug['reward_idxs'] = reward_idxs
            self._tf_debug['reward_idxs_flat'] = reward_idxs_flat
            self._tf_debug['tf_rewards'] = tf_rewards
            self._tf_debug['tf_values'] = tf_values

            mse = tf.reduce_mean(tf.square(tf_values_ph +
                                           tf_target_mask_ph * np.power(self._gamma, self._H)*tf_target_values_max -
                                           tf_values))
            weight_decay = self._weight_decay * tf.add_n(tf.get_collection('weight_decays'))
            cost = mse + weight_decay

        return cost, mse

    ################
    ### Training ###
    ################

    def update_preprocess(self, preprocess_stats):
        obs_mean, obs_orth,  rewards_mean, rewards_orth = \
            preprocess_stats['observations_mean'], \
            preprocess_stats['observations_orth'], \
            preprocess_stats['rewards_mean'], \
            preprocess_stats['rewards_orth']

        self._tf_sess.run([
            self._d_preprocess['observations_mean_assign'],
            # self._d_preprocess['observations_orth_assign'],
            self._d_preprocess['rewards_mean_assign'],
            # self._d_preprocess['rewards_orth_assign']
        ],
            feed_dict={
                self._d_preprocess['observations_mean_ph']: obs_mean,
                self._d_preprocess['rewards_mean_ph']: np.expand_dims(rewards_mean * self._H, 0), # b/c sum of rewards
            })

    def train_step(self, observations, actions, rewards, dones, use_target):
        batch_size = len(observations)
        action_dim = self._env_spec.action_space.flat_dim

        policy_observations = observations[:, 0, :]
        policy_actions = actions[:, :self._H, :].reshape((batch_size, self._H * action_dim))
        policy_rewards = rewards[:, :self._H]
        target_observations = observations[:, self._H, :]

        feed_dict = {
            ### policy
            self._tf_obs_ph: policy_observations,
            self._tf_actions_ph: policy_actions,
            self._tf_rewards_ph: policy_rewards,
            ### target network
            self._tf_obs_target_ph: target_observations,
            self._tf_target_mask_ph: float(use_target) * (1 - dones[:, self._H].astype(float))
        }
        cost, _ = self._tf_sess.run([self._tf_cost, self._tf_opt], feed_dict=feed_dict)

        assert (np.isfinite(cost))

        # debug_keys = sorted(self._tf_debug.keys())
        # eval_debug = self._tf_sess.run([self._tf_debug[k] for k in debug_keys], feed_dict=feed_dict)
        # d = {k: v for (k, v) in zip(debug_keys, eval_debug)}
        # import IPython; IPython.embed()

        self._log_stats['Cost'].append(cost)
        for k, v in self._log_stats.items():
            if len(v) > self._log_history_len:
                self._log_stats[k] = v[1:]

    ######################
    ### Policy methods ###
    ######################

    def get_actions(self, observations, return_action_info=False):
        action_dim = self._env_spec.action_space.flat_dim
        num_obs = len(observations)
        observations = self._env_spec.observation_space.flatten_n(observations)

        pred_values = self._tf_sess.run(self._tf_values,
                                        feed_dict={self._tf_obs_ph: observations})

        chosen_actions = []
        for i, (observation_i, pred_values_i) in enumerate(zip(observations, pred_values)):
            def baseN(num, b, numerals="0123456789abcdefghijklmnopqrstuvwxyz"):
                return ((num == 0) and numerals[0]) or (baseN(num // b, b, numerals).lstrip(numerals[0]) + numerals[num % b])

            chosen_index = int(pred_values_i.argmax())
            chosen_action_i = int(baseN(chosen_index, action_dim)[-1])

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
                'values': pred_values
            }
        else:
            action_info = dict()

        return chosen_actions, action_info

