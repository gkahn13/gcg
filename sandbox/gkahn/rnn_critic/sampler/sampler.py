import pickle
import numpy as np

from sandbox.rocky.tf.envs.vec_env_executor import VecEnvExecutor

from sandbox.gkahn.rnn_critic.sampler.replay_pool import RNNCriticReplayPool

class RNNCriticSampler(object):
    def __init__(self, policy, env, n_envs, replay_pool_size, max_path_length):
        self._policy = policy
        self._n_envs = n_envs

        self._replay_pools = [RNNCriticReplayPool(env.spec,
                                                  policy.H,
                                                  replay_pool_size // n_envs,
                                                  log_history_len=(10 * max_path_length) // n_envs)
                              for _ in range(n_envs)]
        self._vec_env = VecEnvExecutor(
            envs=[pickle.loads(pickle.dumps(env)) for _ in range(self._n_envs)],
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

