import os, pickle, joblib
import itertools
import numpy as np

from rllab.misc.ext import get_seed
from rllab.envs.gym_env import GymEnv
try:
    import gym_ple
except:
    pass

from sandbox.rocky.tf.envs.vec_env_executor import VecEnvExecutor

from sandbox.gkahn.rnn_critic.sampler.replay_pool import RNNCriticReplayPool
from sandbox.gkahn.rnn_critic.utils import utils
from sandbox.gkahn.rnn_critic.policies.mac_policy import MACPolicy
from sandbox.gkahn.rnn_critic.envs.single_env_executor import SingleEnvExecutor
from sandbox.gkahn.rnn_critic.utils.utils import timeit

class RNNCriticSampler(object):
    def __init__(self, policy, env, n_envs, replay_pool_size, max_path_length,
                 save_rollouts=False, save_rollouts_observations=True):
        self._policy = policy
        self._n_envs = n_envs

        self._replay_pools = [RNNCriticReplayPool(env.spec,
                                                  policy.N,
                                                  replay_pool_size // n_envs,
                                                  obs_history_len=policy.obs_history_len,
                                                  save_rollouts=save_rollouts,
                                                  save_rollouts_observations=save_rollouts_observations)
                              for _ in range(n_envs)]
        if self._n_envs > 1:
            envs = [pickle.loads(pickle.dumps(env)) for _ in range(self._n_envs)]
            ### need to seed each environment if it is GymEnv
            seed = get_seed()
            if seed is not None and isinstance(utils.inner_env(env), GymEnv):
                for i, env in enumerate(envs):
                    utils.inner_env(env).env.seed(seed + i)
                self._vec_env = VecEnvExecutor(
                    envs=envs,
                    max_path_length=max_path_length
                )
            self._curr_observations = self._vec_env.reset()
        else:
            self._vec_env = env
            self._curr_observations = [self._vec_env.reset()]
            self._max_path_length = max_path_length
            self._curr_path_length = 0

    @property
    def n_envs(self):
        return self._n_envs

    ##################
    ### Statistics ###
    ##################

    @property
    def statistics(self):
        return RNNCriticReplayPool.statistics_pools(self._replay_pools)

    ####################
    ### Add to pools ###
    ####################

    def step(self, step, take_random_actions=False):
        """ Takes one step in each simulator and adds to respective replay pools """
        ### store last observations and get encoded
        encoded_observations = []
        for i, (replay_pool, observation) in enumerate(zip(self._replay_pools, self._curr_observations)):
            replay_pool.store_observation(step + i, observation)
            encoded_observations.append(replay_pool.encode_recent_observation())

        ### get actions and take step
        if take_random_actions:
            actions = [self._vec_env.action_space.sample() for _ in range(self._n_envs)]
        else:
            actions, _ = self._policy.get_actions(encoded_observations)
        if self._n_envs == 1:
            actions = actions[0]
        next_observations, rewards, dones, _ = self._vec_env.step(actions)
        if self._n_envs == 1:
            self._curr_path_length += 1
            dones = dones or self._curr_path_length > self._max_path_length
            if dones:
                next_observations = self._vec_env.reset()
                self._curr_path_length = 0
            actions = [actions]
            next_observations = [next_observations]
            rewards = [rewards]
            dones = [dones]

        ### add to replay pool
        for replay_pool, action, reward, done in zip(self._replay_pools, actions, rewards, dones):
            replay_pool.store_effect(action, reward, done)
        self._curr_observations = next_observations

    #####################
    ### Add offpolicy ###
    #####################

    def _offpolicy_itr_file(self, offpolicy_folder, itr):
        return os.path.join(offpolicy_folder, 'itr_{0:d}.pkl'.format(itr))

    def add_offpolicy(self, offpolicy_folder):
        step = 0
        itr = 0
        replay_pools = itertools.cycle(self._replay_pools)

        while os.path.exists(self._offpolicy_itr_file(offpolicy_folder, itr)):
            sess, graph = MACPolicy.create_session_and_graph(gpu_device='')
            with graph.as_default(), sess.as_default():
                d = joblib.load(self._offpolicy_itr_file(offpolicy_folder, itr))
                rollouts = d['rollouts']
                d['policy'].terminate()
            itr += 1

            for rollout, replay_pool in zip(rollouts, replay_pools):
                replay_pool.store_rollout(step, rollout)
                step += len(rollout['dones'])

    #########################
    ### Sample from pools ###
    #########################

    def can_sample(self):
        return np.any([replay_pool.can_sample() for replay_pool in self._replay_pools])

    def sample(self, batch_size):
        return RNNCriticReplayPool.sample_pools(self._replay_pools, batch_size)

    ###############
    ### Logging ###
    ###############

    def log(self):
        RNNCriticReplayPool.log_pools(self._replay_pools)

    def get_recent_paths(self):
        return RNNCriticReplayPool.get_recent_paths_pools(self._replay_pools)

