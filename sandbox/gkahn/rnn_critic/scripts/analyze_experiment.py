import argparse, os, sys, copy
import yaml, pickle
import joblib
import itertools
import pandas
import time

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import ticker
from sklearn.utils.extmath import cartesian
import tensorflow as tf

from rllab.misc.ext import set_seed
import rllab.misc.logger as logger
# from rllab.misc import tensor_utils
from sandbox.rocky.tf.misc import tensor_utils

from sandbox.gkahn.rnn_critic.envs.point_env import PointEnv
from sandbox.gkahn.rnn_critic.envs.sparse_point_env import SparsePointEnv
from sandbox.gkahn.rnn_critic.envs.chain_env import ChainEnv
from sandbox.gkahn.rnn_critic.envs.phd_env import PhdEnv
from sandbox.gkahn.rnn_critic.policies.mac_policy import MACPolicy
from sandbox.gkahn.rnn_critic.sampler.vectorized_rollout_sampler import RNNCriticVectorizedRolloutSampler
from sandbox.gkahn.rnn_critic.sampler.sampler import RNNCriticSampler

### environments
from sandbox.gkahn.rnn_critic.envs.env_utils import create_env

#########################
### Utility functions ###
#########################

def rollout_policy(env, agent, max_path_length=np.inf, animated=False, speedup=1, start_obs=None):
    observations = []
    actions = []
    rewards = []
    agent_infos = []
    env_infos = []
    if start_obs is None:
        o = env.reset()
    else:
        o = start_obs
    agent.reset()
    path_length = 0
    if animated:
        env.render()
    while path_length < max_path_length:
        a, agent_info = agent.get_action(step=path_length, observation=o, explore=False)
        next_o, r, d, env_info = env.step(a)
        observations.append(env.observation_space.flatten(o))
        rewards.append(r)
        actions.append(env.action_space.flatten(a))
        agent_infos.append(agent_info)
        if isinstance(inner_env(env), GymEnv):
            ienv = inner_env(env).env.env.env
            if hasattr(ienv, 'model'):
                env_info['qpos'] = ienv.model.data.qpos
            if hasattr(ienv, 'state'):
                env_info['state'] = ienv.state
        env_infos.append(env_info)
        path_length += 1
        if d:
            break
        o = next_o
        if animated:
            env.render()
            timestep = 0.05
            time.sleep(timestep / speedup)
    if animated:
        return

    return dict(
        observations=tensor_utils.stack_tensor_list(observations),
        actions=tensor_utils.stack_tensor_list(actions),
        rewards=tensor_utils.stack_tensor_list(rewards),
        agent_infos=tensor_utils.stack_tensor_dict_list(agent_infos),
        env_infos=tensor_utils.stack_tensor_dict_list(env_infos),
    )

def inner_env(env):
    while hasattr(env, 'wrapped_env'):
        env = env.wrapped_env
    return env

def moving_avg_std(idxs, data, window):
    avg_idxs, means, stds = [], [], []
    for i in range(window, len(data)):
        avg_idxs.append(np.mean(idxs[i - window:i]))
        means.append(np.mean(data[i - window:i]))
        stds.append(np.std(data[i - window:i]))
    return avg_idxs, np.asarray(means), np.asarray(stds)

################
### Analysis ###
################

class AnalyzeRNNCritic(object):
    def __init__(self, folder, skip_itr=1, max_itr=sys.maxsize, plot=dict(), create_new_envs=True, clear_obs=False, load_eval_rollouts=True, load_train_rollouts=True):
        """
        :param kwargs: holds random extra properties
        """
        self._folder = folder
        self._skip_itr = skip_itr
        self._max_itr = max_itr
        self._create_new_envs = create_new_envs
        self._clear_obs = clear_obs

        ### load data
        # logger.log('AnalyzeRNNCritic: Loading data')
        self.name = os.path.basename(self._folder)
        self.plot = plot
        # logger.log('AnalyzeRNNCritic: params_file: {0}'.format(self._params_file))
        with open(self._params_file, 'r') as f:
            self.params = yaml.load(f)
        # logger.log('AnalyzeRNNCritic: Loading csv')
        try:
            self.progress = pandas.read_csv(self._progress_file)
        except Exception as e:
            logger.log('Could not open csv: {0}'.format(str(e)))
            self.progress = None
        # logger.log('AnalyzeRNNCritic: Loaded csv')

        self.train_rollouts_itrs = self._load_rollouts_itrs() if load_train_rollouts else None
        self.eval_rollouts_itrs = self._load_rollouts_itrs(eval=True) if load_eval_rollouts else None

        # logger.log('AnalyzeRNNCritic: Loaded all itrs')
        if create_new_envs:
            self.env = create_env(self.params['alg']['env'])
        else:
            self.env = None
        # logger.log('AnalyzeRNNCritic: Created env')
        # logger.log('AnalyzeRNNCritic: Finished loading data')

    #############
    ### Files ###
    #############

    def _itr_file(self, itr):
        return os.path.join(self._folder, 'itr_{0:d}.pkl'.format(itr))

    def _itr_rollouts_file(self, itr, eval=False):
        if eval:
            fname = 'itr_{0}_rollouts_eval.pkl'.format(itr)
        else:
            fname = 'itr_{0}_rollouts.pkl'.format(itr)
        return os.path.join(self._folder, fname)

    @property
    def _progress_file(self):
        return os.path.join(self._folder, 'progress.csv')

    @property
    def _eval_rollouts_itrs_file(self):
        return os.path.join(self._folder, 'eval_rollouts_itrs.pkl')

    @property
    def _params_file(self):
        yamls = [fname for fname in os.listdir(self._folder) if os.path.splitext(fname)[-1] == '.yaml' and os.path.basename(self._folder) in fname]
        assert(len(yamls) == 1)
        return os.path.join(self._folder, yamls[0])

    ####################
    ### Data loading ###
    ####################

    def _load_rollouts_itrs(self, eval=False):
        train_rollouts_itrs = []
        itr = 0
        while os.path.exists(self._itr_rollouts_file(itr, eval=eval)) and itr < self._max_itr:
            rollouts = joblib.load(self._itr_rollouts_file(itr, eval=eval))['rollouts']
            train_rollouts_itrs.append(rollouts)
            itr += self._skip_itr

        return train_rollouts_itrs

    def _load_itr_policy(self, itr):
        d = joblib.load(self._itr_file(itr))
        policy = d['policy']
        return policy

    def eval_policy(self, itr, gpu_device=None, gpu_frac=None):
        if self.params['seed'] is not None:
            set_seed(self.params['seed'])
            try:
                inner_env(self.env).env.seed(self.params['seed'])
            except:
                pass

        if gpu_device is None:
            gpu_device = self.params['policy']['gpu_device']
        if gpu_frac is None:
            gpu_frac = self.params['policy']['gpu_frac']
        sess, graph = MACPolicy.create_session_and_graph(gpu_device=gpu_device, gpu_frac=gpu_frac)
        with graph.as_default(), sess.as_default():
            policy = self._load_itr_policy(itr)

            logger.log('Evaluating policy for itr {0}'.format(itr))
            n_envs = 1
            if 'max_path_length' in self.params['alg']:
                max_path_length = self.params['alg']['max_path_length']
            else:
                max_path_length = self.env.horizon

            import IPython; IPython.embed()

            sampler = RNNCriticSampler(
                policy=policy,
                env=self.env,
                n_envs=n_envs,
                replay_pool_size=int(1e4),
                max_path_length=max_path_length,
                save_rollouts=True,
                sampling_method=self.params['alg']['replay_pool_sampling']
            )
            rollouts = []
            step = 0
            while len(rollouts) < 25:
                sampler.step(step)
                step += n_envs
                rollouts += sampler.get_recent_paths()

        return rollouts
