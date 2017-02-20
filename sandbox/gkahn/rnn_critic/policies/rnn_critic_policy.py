import inspect
import time
import os
from collections import defaultdict
import itertools

import numpy as np
import scipy.linalg
import tensorflow as tf
from sklearn.utils.extmath import cartesian

from rllab.core.serializable import Serializable
import rllab.misc.logger as logger

from sandbox.rocky.tf.policies.base import Policy
from sandbox.gkahn.tf.core.parameterized import Parameterized

class RNNCriticPolicy(Policy, Parameterized, Serializable):
    def __init__(self,
                 is_train,
                 env_spec,
                 H,
                 weight_decay,
                 learning_rate,
                 reset_every_train,
                 train_steps,
                 batch_size,
                 get_action_params,
                 gpu_device=None,
                 gpu_frac=None,
                 **kwargs):
        """
        :param is_train: if True file placeholders, else feed placeholders
        :param H: critic horizon length
        :param weight_decay
        :param learning_rate
        :param reset_every_train: reset parameters every time train is called?
        :param train_steps: how many calls to optimizer each time train is called
        :param batch_size
        :param get_action_params: dictionary specifying how to choose actions
        """
        Serializable.quick_init(self, locals())

        self._is_train = is_train
        self._env_spec = env_spec
        self._H = H
        self._weight_decay = weight_decay
        self._learning_rate = learning_rate
        self._reset_every_train = reset_every_train
        self._train_steps = train_steps
        self._batch_size = batch_size
        self._get_action_params = get_action_params
        self._gpu_device = gpu_device
        self._gpu_frac = gpu_frac
        self._exploration_strategy = None # don't set in init b/c will then be Serialized

        # for attr in [attr for attr in dir(self) if '__' not in attr and not inspect.ismethod(getattr(self, attr))]:
        #     logger.log('RNNCriticPolicy\t{0}: {1}'.format(attr, getattr(self, attr)))

        self._tf_graph, self._tf_sess, \
            self._tf_obs_ph, self._tf_actions_ph, self._tf_rewards_ph, self._d_queue, self._d_preprocess, \
            self._tf_rewards, self._tf_cost, self._tf_opt = self._graph_setup()

        self._get_action_preprocess = self._get_action_setup()

        Policy.__init__(self, env_spec)
        Parameterized.__init__(self, sess=self._tf_sess)

    ##################
    ### Properties ###
    ##################

    @property
    def H(self):
        return self._H

    @property
    def session(self):
        return self._tf_sess

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
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_frac)
        config = tf.ConfigProto(gpu_options=gpu_options,
                                log_device_placement=False,
                                allow_soft_placement=True)
        tf_sess = tf.Session(graph=tf_graph, config=config)
        return tf_sess, tf_graph

    def _graph_inputs_outputs_from_placeholders(self):
        with tf.variable_scope('feed_input'):
            tf_obs_ph = tf.placeholder('float', [None, self._env_spec.observation_space.flat_dim])
            tf_actions_ph = tf.placeholder('float', [None, self._env_spec.action_space.flat_dim * self._H])
            tf_rewards_ph = tf.placeholder('float', [None, self._H])

        return tf_obs_ph, tf_actions_ph, tf_rewards_ph, dict()

    def _graph_inputs_outputs_from_files(self):
        obs_dim = self._env_spec.observation_space.flat_dim
        action_dim = self._env_spec.action_space.flat_dim

        with tf.name_scope('file_input'):
            queue_ph = tf.placeholder(tf.string)
            queue_var = tf.get_variable('fnames', initializer=queue_ph, validate_shape=False, trainable=False)

            ### create file queue
            filename_queue = tf.train.string_input_producer(queue_var, num_epochs=None, shuffle=True)

            ### read and decode
            reader = tf.TFRecordReader()
            _, serialized_example = reader.read(filename_queue)
            features = {
                'observation': tf.FixedLenFeature([obs_dim], tf.float32),
                'actions': tf.FixedLenFeature([action_dim * self._H], tf.float32),
                'rewards': tf.FixedLenFeature([self._H], tf.float32)
            }
            parsed_example = tf.parse_single_example(serialized_example, features=features)

            single_observation = parsed_example['observation']
            single_actions = parsed_example['actions']
            single_rewards = parsed_example['rewards']

            ### shuffle and put into batches
            tf_obs_ph, tf_actions_ph, tf_rewards_ph = tf.train.shuffle_batch(
                [single_observation, single_actions, single_rewards], batch_size=self._batch_size, num_threads=2,
                capacity=1000 + 3 * self._batch_size,
                # Ensures a minimum amount of shuffling of examples.
                min_after_dequeue=1000)

            d_queue = {
                'ph': queue_ph,
                'var': queue_var,
                'queue': filename_queue
            }

        return tf_obs_ph, tf_actions_ph, tf_rewards_ph, d_queue

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

        num_obs = tf.shape(tf_obs_whitened)[0]
        num_action = tf.shape(tf_actions_whitened)[0]

        def tf_repeat_2d(x, reps):
            """ Repeats x on axis=0 reps times """
            x_shape = tf.shape(x)
            x_repeat = tf.reshape(tf.tile(x, [1, reps]), (x_shape[0] * reps, x_shape[1]))
            x_repeat.set_shape((None, x.get_shape()[1]))
            return x_repeat

        ### replicate observation for each action
        def replicate_observation():
            tf_obs_whitened_rep = tf_repeat_2d(tf_obs_whitened, num_action // num_obs)
            # tf_obs_whitened_tiled = tf.tile(tf_obs_whitened, tf.pack([batch_size, 1]))
            # tf_obs_whitened_tiled.set_shape([None, tf_obs_whitened.get_shape()[1]])
            return tf_obs_whitened_rep

        # assumes num_action is a multiple of num_obs
        tf_obs_whitened_cond = tf.cond(tf.not_equal(num_obs, num_action), replicate_observation, lambda: tf_obs_whitened)

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

    def _graph_init_vars(self, tf_sess, d_queue):
        feed_dict = dict()
        if 'ph' in d_queue:
            feed_dict[d_queue['ph']] = []
        tf_sess.run([tf.initialize_all_variables()], feed_dict=feed_dict)

    def _graph_setup(self):
        tf_sess = tf.get_default_session()
        if tf_sess is None:
            tf_sess, tf_graph = RNNCriticPolicy.create_session_and_graph(gpu_device=self._gpu_device,
                                                                         gpu_frac=self._gpu_frac)
        else:
            tf_graph = tf_sess.graph

        with tf_graph.as_default():
            if self._is_train:
                tf_obs_ph, tf_actions_ph, tf_rewards_ph, d_queue = self._graph_inputs_outputs_from_files()
            else:
                tf_obs_ph, tf_actions_ph, tf_rewards_ph, d_queue = self._graph_inputs_outputs_from_placeholders()
            pre_variables = tf.all_variables()
            d_preprocess = self._graph_preprocess_from_placeholders()
            tf_rewards = self._graph_inference(tf_obs_ph, tf_actions_ph, d_preprocess)
            post_variables = tf.all_variables()
            for v in set(post_variables).difference(pre_variables):
                tf.add_to_collection('params_internal', v)
            if self._is_train:
                tf_cost, tf_mse = self._graph_cost(tf_rewards_ph, tf_rewards)
                tf_opt = self._graph_optimize(tf_cost)
            else:
                tf_cost, tf_mse, tf_opt = None, None, None

            self._graph_init_vars(tf_sess, d_queue)

            merged = tf.merge_all_summaries()
            writer = tf.train.SummaryWriter('/tmp', graph_def=tf_sess.graph_def)

        return tf_graph, tf_sess, tf_obs_ph, tf_actions_ph, tf_rewards_ph, d_queue, d_preprocess, tf_rewards, tf_cost, tf_opt

    def _graph_set_preprocess(self, preprocess_stats):
        obs_mean, obs_orth, actions_mean, actions_orth, rewards_mean, rewards_orth = \
            preprocess_stats['observations_mean'], \
            preprocess_stats['observations_orth'], \
            preprocess_stats['actions_mean'], \
            preprocess_stats['actions_orth'], \
            preprocess_stats['rewards_mean'], \
            preprocess_stats['rewards_orth']
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

    def _graph_reset_queue(self, tfrecords, flush=True):
        # new file inputs
        self._tf_sess.run([tf.assign(self._d_queue['var'], tfrecords, validate_shape=False)])
        # flush out old ones
        if flush:
            self._tf_sess.run(self._d_queue['queue'].dequeue_many(2000 + 3 * self._batch_size)) # TODO: hard coded

    def train(self, tfrecords, preprocess_stats):
        """
        :type replay_pool: RNNCriticReplayPool
        """
        assert(self._is_train)
        start_time = time.time()

        with self._tf_graph.as_default(), self._tf_sess.as_default():
            if self._reset_every_train:
                self._graph_init_vars(self._tf_sess, self._d_queue)

            if not hasattr(self, '_tf_coord'):
                assert (not hasattr(self, '_tf_threads'))
                self._graph_reset_queue(tfrecords, flush=False)
                self._tf_coord = tf.train.Coordinator()
                self._tf_threads = tf.train.start_queue_runners(sess=self._tf_sess, coord=self._tf_coord)

            self._graph_set_preprocess(preprocess_stats)
            self._graph_reset_queue(tfrecords)

            train_log = defaultdict(list)

            for step in range(self._train_steps):
                cost, _ = self._tf_sess.run([self._tf_cost, self._tf_opt])
                train_log['cost'].append(cost)

        train_log['TrainTime'] = time.time() - start_time
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

    def set_exploration_strategy(self, exploration_strategy):
        self._exploration_strategy = exploration_strategy

    def get_action(self, observation):
        assert(not self._is_train)

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

        if self._exploration_strategy is not None:
            exploration_func = lambda: None
            exploration_func.get_action = lambda _: (chosen_action, dict())
            chosen_action = self._exploration_strategy.get_action(0, observation, exploration_func)

        return chosen_action, dict()

    def get_actions(self, observations):
        assert(not self._is_train)

        num_obs = len(observations)
        action_lower, action_upper = self._env_spec.action_space.bounds

        if self._get_action_params['type'] == 'random':
            N = self._get_action_params['N']
            actions = np.random.uniform(action_lower.tolist(), action_upper.tolist(),
                                        size=(N * num_obs, self._H, self._env_spec.action_space.flat_dim))
            actions = actions.reshape(N * num_obs, self._H * self._env_spec.action_space.flat_dim)
        else:
            raise NotImplementedError('get_actions type {0} not implemented'.format(self._get_action_params['type']))

        pred_rewards = self._tf_sess.run([self._tf_rewards],
                                         feed_dict={self._tf_obs_ph: observations,
                                                    self._tf_actions_ph: actions})[0]

        chosen_actions = []
        for observation_i, pred_rewards_i, actions_i in zip(observations,
                                                           np.split(pred_rewards, num_obs, axis=0),
                                                           np.split(actions, num_obs, axis=0)):
            chosen_action_i = actions_i[pred_rewards_i.sum(axis=1).argmax()][:self._env_spec.action_space.flat_dim]
            if self._exploration_strategy is not None:
                exploration_func = lambda: None
                exploration_func.get_action = lambda _: (chosen_action_i, dict())
                chosen_action_i = self._exploration_strategy.get_action(0, observation_i, exploration_func)
            chosen_actions.append(chosen_action_i)

        return chosen_actions, dict()

    @property
    def recurrent(self):
        raise NotImplementedError

    def terminate(self):
        self._tf_sess.close()

    ######################
    ### Saving/loading ###
    ######################

    def get_params_internal(self, **tags):
        return sorted(self._tf_graph.get_collection('params_internal'), key=lambda v: v.name)

    def match(self, other_policy):
        """ Update self to match the other policy """
        assert(type(self) == type(other_policy))

        ### get params
        self_params = self.get_params_internal()
        other_params = other_policy.get_params_internal()

        ### make sure params are the same
        self_params_names = [p.name for p in self_params]
        other_params_names = [p.name for p in other_params]
        assert(self_params_names == other_params_names)

        ### evaluate other params and set self params
        other_params_values = other_policy._tf_sess.run(other_params)
        self._tf_sess.run([tf.assign(p, v) for p, v in zip(self_params, other_params_values)])
