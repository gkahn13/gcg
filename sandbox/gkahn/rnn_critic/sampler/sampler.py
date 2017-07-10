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
from sandbox.rocky.tf.spaces.discrete import Discrete
from sandbox.rocky.tf.spaces.box import Box

class RNNCriticSampler(object):
    def __init__(self, policy, env, n_envs, replay_pool_size, max_path_length, sampling_method,
                 save_rollouts=False, save_rollouts_observations=True):
        self._policy = policy
        self._n_envs = n_envs

        self._replay_pools = [RNNCriticReplayPool(env.spec,
                                                  policy.N,
                                                  policy.gamma,
                                                  replay_pool_size // n_envs,
                                                  obs_history_len=policy.obs_history_len,
                                                  sampling_method=sampling_method,
                                                  save_rollouts=save_rollouts,
                                                  save_rollouts_observations=save_rollouts_observations)
                              for _ in range(n_envs)]

        envs = [pickle.loads(pickle.dumps(env)) for _ in range(self._n_envs)] if self._n_envs > 1 else [env]
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

    def step(self, step, take_random_actions=False, explore=True):
        """ Takes one step in each simulator and adds to respective replay pools """
        ### store last observations and get encoded
        encoded_observations = []
        for i, (replay_pool, observation) in enumerate(zip(self._replay_pools, self._curr_observations)):
            replay_pool.store_observation(step + i, observation)
            encoded_observations.append(replay_pool.encode_recent_observation())

        ### get actions
        if take_random_actions:
            actions = [self._vec_env.action_space.sample() for _ in range(self._n_envs)]
            est_values = [np.nan] * self._n_envs
            if isinstance(self._vec_env.action_space, Discrete):
                logprobs = [-np.log(self._vec_env.action_space.flat_dim)] * self._n_envs
            elif isinstance(self._vec_env.action_space, Box):
                low = self._vec_env.action_space.low
                high = self._vec_env.action_space.high
                logprobs = [-np.sum(np.log(high - low))] * self._n_envs
            else:
                raise NotImplementedError
        else:
            actions, est_values, logprobs, _ = self._policy.get_actions(list(range(step, step + self._n_envs)),
                                                                        encoded_observations,
                                                                        explore=explore)

        ### take step
        next_observations, rewards, dones, _ = self._vec_env.step(actions)

        ### add to replay pool
        for replay_pool, action, reward, done, est_value, logprob in \
                zip(self._replay_pools, actions, rewards, dones, est_values, logprobs):
            replay_pool.store_effect(action, reward, done, est_value, logprob)

        self._curr_observations = next_observations

    #####################
    ### Add offpolicy ###
    #####################

    def _rollouts_file(self, folder, itr):
        return os.path.join(folder, 'itr_{0:d}_rollouts.pkl'.format(itr))

    def add_offpolicy(self, offpolicy_folder, num_offpolicy):
        step = 0
        itr = 0
        replay_pools = itertools.cycle(self._replay_pools)
        done_adding = False

        while os.path.exists(self._rollouts_file(offpolicy_folder, itr)):
            rollouts = joblib.load(self._rollouts_file(offpolicy_folder, itr))['rollouts']
            itr += 1

            for rollout, replay_pool in zip(rollouts, replay_pools):
                r_len = len(rollout['dones'])
                if step + r_len >= num_offpolicy:
                    diff = num_offpolicy - step
                    for k in ('observations', 'actions', 'rewards', 'dones', 'logprobs'):
                        rollout[k] = rollout[k][:diff]
                    done_adding = True
                    r_len = len(rollout['dones'])

                replay_pool.store_rollout(step, rollout)
                step += r_len

                if done_adding:
                    break

            if done_adding:
                break

    #########################
    ### Sample from pools ###
    #########################

    def can_sample(self):
        return np.any([replay_pool.can_sample() for replay_pool in self._replay_pools])

    def sample(self, batch_size):
        return RNNCriticReplayPool.sample_pools(self._replay_pools, batch_size,
                                                only_completed_episodes=self._policy.only_completed_episodes)

    ###############
    ### Logging ###
    ###############

    def log(self, prefix=''):
        RNNCriticReplayPool.log_pools(self._replay_pools, prefix=prefix)

    def get_recent_paths(self):
        return RNNCriticReplayPool.get_recent_paths_pools(self._replay_pools)

