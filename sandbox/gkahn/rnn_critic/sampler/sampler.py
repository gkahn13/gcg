import pickle
import numpy as np

from rllab.misc.ext import get_seed
from rllab.envs.gym_env import GymEnv
import gym_ple

from sandbox.rocky.tf.envs.vec_env_executor import VecEnvExecutor

from sandbox.gkahn.rnn_critic.sampler.replay_pool import RNNCriticReplayPool
from sandbox.gkahn.rnn_critic.utils import utils

class RNNCriticSampler(object):
    def __init__(self, policy, env, n_envs, replay_pool_size, max_path_length, save_rollouts=False):
        self._policy = policy
        self._n_envs = n_envs

        self._replay_pools = [RNNCriticReplayPool(env.spec,
                                                  policy.H,
                                                  replay_pool_size // n_envs,
                                                  save_rollouts=save_rollouts)
                              for _ in range(n_envs)]
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

    def step(self, step):
        """ Takes one step in each simulator and adds to respective replay pools """
        actions, _ = self._policy.get_actions(self._curr_observations)
        next_observations, rewards, dones, _ = self._vec_env.step(actions)
        for i, replay_pool in enumerate(self._replay_pools):
            replay_pool.add(step + i,
                            self._curr_observations[i],
                            actions[i],
                            rewards[i],
                            dones[i])
        self._curr_observations = next_observations

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

