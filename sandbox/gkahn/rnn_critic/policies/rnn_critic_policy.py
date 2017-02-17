import inspect
from collections import defaultdict
import itertools

import numpy as np
import scipy.linalg
import tensorflow as tf
from sklearn.utils.extmath import cartesian

from rllab.core.serializable import Serializable
import rllab.misc.logger as logger

from sandbox.rocky.tf.policies.base import Policy
from sandbox.rocky.tf.core.parameterized import Parameterized

class RNNCriticPolicy(Policy, Parameterized, Serializable):
    def __init__(self,
                 env_spec,
                 H,
                 weight_decay,
                 learning_rate,
                 reset_every_train,
                 train_steps,
                 batch_size,
                 get_action_params):
        """
        :param H: critic horizon length
        :param weight_decay
        :param learning_rate
        :param reset_every_train: reset parameters every time train is called?
        :param train_steps: how many calls to optimizer each time train is called
        :param batch_size
        :param get_action_params: dictionary specifying how to choose actions
        """
        Serializable.quick_init(self, locals())

        self._env_spec = env_spec
        self._H = H
        self._weight_decay = weight_decay
        self._learning_rate = learning_rate
        self._reset_every_train = reset_every_train
        self._train_steps = train_steps
        self._batch_size = batch_size
        self._get_action_params = get_action_params

        for attr in [attr for attr in dir(self) if '__' not in attr and not inspect.ismethod(getattr(self, attr))]:
            logger.log('RNNCriticPolicy\t{0}: {1}'.format(attr, getattr(self, attr)))

        self._tf_obs_ph, self._tf_actions_ph, self._tf_rewards_ph, self._d_preprocess, \
            self._tf_rewards, self._tf_cost, self._tf_opt, self._tf_sess, self._tf_saver = self._graph_setup()

        self._get_action_preprocess = self._get_action_setup()

        # super(RNNCriticPolicy, self).__init__(env_spec)
        Policy.__init__(self, env_spec)
        # Parameterized.__init__(self, sess=self._tf_sess)

    ###########################
    ### TF graph operations ###
    ###########################

    def _graph_inputs_outputs_from_placeholders(self):
        with tf.variable_scope('feed_input'):
            tf_obs_ph = tf.placeholder('float', [None, self._env_spec.observation_space.flat_dim])
            tf_actions_ph = tf.placeholder('float', [None, self._env_spec.action_space.flat_dim * self._H])
            tf_rewards_ph = tf.placeholder('float', [None, self._H])

        return tf_obs_ph, tf_actions_ph, tf_rewards_ph

    def _graph_preprocess_from_placeholders(self):
        d_preprocess = dict()

        with tf.variable_scope('preprocess'):
            for name, dim in (('observations', self._env_spec.observation_space.flat_dim),
                              ('actions', self._env_spec.action_space.flat_dim * self._H),
                              ('rewards', self._H)):
                d_preprocess[name+'_mean_ph'] = tf.placeholder(tf.float32, shape=(1, dim))
                d_preprocess[name+'_mean_var'] = tf.get_variable(name+'_mean_var', shape=[1, dim],
                                                                 trainable=False, dtype=tf.float32,
                                                                 initializer=tf.constant_initializer(np.zeros((1, dim))))
                d_preprocess[name+'_mean_assign'] = tf.assign(d_preprocess[name+'_mean_var'],
                                                              d_preprocess[name+'_mean_ph'])

                d_preprocess[name+'_orth_ph'] = tf.placeholder(tf.float32, shape=(dim, dim))
                d_preprocess[name+'_orth_var'] = tf.get_variable(name+'_orth_var',
                                                                 shape=(dim, dim),
                                                                 trainable=False, dtype=tf.float32,
                                                                 initializer=tf.constant_initializer(np.eye(dim)))
                d_preprocess[name+'_orth_assign'] = tf.assign(d_preprocess[name+'_orth_var'],
                                                              d_preprocess[name+'_orth_ph'])

        return d_preprocess

    def _graph_preprocess_inputs(self, tf_obs_ph, tf_actions_ph, d_preprocess):
        ### whiten inputs
        tf_obs_whitened = tf.matmul(tf_obs_ph - d_preprocess['observations_mean_var'],
                                    d_preprocess['observations_orth_var'])
        tf_actions_whitened = tf.matmul(tf_actions_ph - d_preprocess['actions_mean_var'],
                                        d_preprocess['actions_orth_var'])

        ### replicate observation for each action
        def replicate_observation():
            batch_size = tf.shape(tf_actions_whitened)[0]
            tf_obs_whitened_tiled = tf.tile(tf_obs_whitened, tf.pack([batch_size, 1]))
            tf_obs_whitened_tiled.set_shape([None, tf_obs_whitened.get_shape()[1]])
            return tf_obs_whitened_tiled

        num_obs = tf.shape(tf_obs_whitened)[0]
        tf_obs_whitened_cond = tf.cond(tf.equal(num_obs, 1), replicate_observation, lambda: tf_obs_whitened)

        return tf_obs_whitened_cond, tf_actions_whitened

    def _graph_preprocess_outputs(self, tf_rewards, d_preprocess):
        return tf.add(tf.matmul(tf_rewards, tf.transpose(d_preprocess['rewards_orth_var'])),
                      d_preprocess['rewards_mean_var'])

    def _graph_inference(self, tf_obs_ph, tf_actions_ph, d_preprocess):
        raise NotImplementedError

    def _graph_cost(self, tf_rewards_ph, tf_rewards):
        mse = tf.reduce_mean(tf.square(tf_rewards_ph - tf_rewards))
        weight_decay = self._weight_decay * tf.add_n(tf.get_collection('weight_decays'))
        cost = mse + weight_decay
        return cost, mse

    def _graph_optimize(self, tf_cost):
        return tf.train.AdamOptimizer(learning_rate=self._learning_rate).minimize(tf_cost)

    def _graph_init_vars(self, tf_sess):
        tf_sess.run([tf.initialize_all_variables()])

    def _graph_setup(self):
        tf_sess = tf.get_default_session()

        tf_obs_ph, tf_actions_ph, tf_rewards_ph = self._graph_inputs_outputs_from_placeholders()
        d_preprocess = self._graph_preprocess_from_placeholders()
        tf_rewards = self._graph_inference(tf_obs_ph, tf_actions_ph, d_preprocess)
        tf_cost, tf_mse = self._graph_cost(tf_rewards_ph, tf_rewards)
        tf_opt = self._graph_optimize(tf_cost)

        self._graph_init_vars(tf_sess)

        merged = tf.merge_all_summaries()
        writer = tf.train.SummaryWriter('/tmp', graph_def=tf_sess.graph_def)

        tf_saver = tf.train.Saver(max_to_keep=None)

        return tf_obs_ph, tf_actions_ph, tf_rewards_ph, d_preprocess, tf_rewards, tf_cost, tf_opt, tf_sess, tf_saver

    def _graph_set_preprocess(self, replay_pool):
        obs_mean, obs_orth, actions_mean, actions_orth, rewards_mean, rewards_orth = replay_pool.get_whitening()
        self._tf_sess.run([
            self._d_preprocess['observations_mean_assign'],
            # self._d_preprocess['observations_orth_assign'],
            self._d_preprocess['actions_mean_assign'],
            # self._d_preprocess['actions_orth_assign'],
            self._d_preprocess['rewards_mean_assign'],
            # self._d_preprocess['rewards_orth_assign']
        ],
          feed_dict={
              self._d_preprocess['observations_mean_ph']: np.expand_dims(obs_mean, 0),
              self._d_preprocess['observations_orth_ph']: obs_orth,
              self._d_preprocess['actions_mean_ph']: np.expand_dims(np.tile(actions_mean, self._H), 0),
              self._d_preprocess['actions_orth_ph']: scipy.linalg.block_diag(*([actions_orth] * self._H)),
              self._d_preprocess['rewards_mean_ph']: np.expand_dims(np.tile(rewards_mean, self._H), 0),
              self._d_preprocess['rewards_orth_ph']: scipy.linalg.block_diag(*([rewards_orth] * self._H))
          })

    def train(self, replay_pool):
        """
        :type replay_pool: RNNCriticReplayPool
        """
        if self._reset_every_train:
            self._graph_init_vars(self._tf_sess)

        self._graph_set_preprocess(replay_pool)

        train_log = defaultdict(list)

        for step in range(self._train_steps):
            observations, actions, rewards = [], [], []
            for b in range(self._batch_size):
                observations_b, actions_b, rewards_b, _ = replay_pool.get_random_sequence(self._H)
                observations.append(observations_b[0])
                actions.append(np.ravel(actions_b))
                rewards.append(rewards_b)

            cost, _ = self._tf_sess.run([self._tf_cost, self._tf_opt],
                                        feed_dict={self._tf_obs_ph: observations,
                                                   self._tf_actions_ph: actions,
                                                   self._tf_rewards_ph: rewards})
            train_log['cost'].append(cost)

        return train_log

    ######################
    ### Policy methods ###
    ######################

    def _get_action_setup(self):
        get_action_preprocess = dict()

        if self._get_action_params['type'] == 'random':
            pass
        elif self._get_action_params['type'] == 'lattice':
            N = self._get_action_params['N']
            action_lower, action_upper = self._env_spec.action_space.bounds
            single_actions = cartesian([np.linspace(l, u, N) for l, u in zip(action_lower, action_upper)])
            actions = np.asarray(list(itertools.combinations(single_actions, self._H)))
            get_action_preprocess['actions'] = actions.reshape((len(actions), self._H * self._env_spec.action_space.flat_dim))
        else:
            raise NotImplementedError('get_action type {0} not implemented'.format(self._get_action_params['type']))

        return get_action_preprocess

    def get_action(self, observation):
        # randomly sample, then evaluate
        action_lower, action_upper = self._env_spec.action_space.bounds

        if self._get_action_params['type'] == 'random':
            N = self._get_action_params['N']
            actions = np.random.uniform(action_lower.tolist(), action_upper.tolist(),
                                        size=(N, self._H, self._env_spec.action_space.flat_dim))
            actions = actions.reshape(N, self._H * self._env_spec.action_space.flat_dim)
        elif self._get_action_params['type'] == 'lattice':
            actions = self._get_action_preprocess['actions']
        else:
            raise NotImplementedError('get_action type {0} not implemented'.format(self._get_action_params['type']))

        pred_rewards = self._tf_sess.run([self._tf_rewards],
                                         feed_dict={self._tf_obs_ph: [observation],
                                                    self._tf_actions_ph: actions})[0]

        chosen_action = actions[pred_rewards.sum(axis=1).argmax()][:self._env_spec.action_space.flat_dim]

        return chosen_action, dict()

    @property
    def recurrent(self):
        raise NotImplementedError

    def terminate(self):
        self._tf_sess.close()

    ######################
    ### Saving/loading ###
    ######################

    def get_params_internal(self, **tags):
        return sorted(tf.all_variables(), key=lambda v: v.name)
