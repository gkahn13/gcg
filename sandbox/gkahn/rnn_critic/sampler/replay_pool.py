import itertools
from collections import defaultdict
import numpy as np

import rllab.misc.logger as logger

class RNNCriticReplayPool(object):

    def __init__(self, env_spec, H, size, log_history_len):
        """
        :param env_spec: for observation/action dimensions
        :param H: horizon length
        :param size: size of pool
        :param log_history_len: length of log history
        """
        self._env_spec = env_spec
        self._H = H
        self._size = size
        self._log_history_len = log_history_len

        ### initialize buffer
        obs_dim = env_spec.observation_space.flat_dim
        action_dim = env_spec.action_space.flat_dim
        self._steps = np.empty((self._size,), dtype=np.int32)
        self._observations = np.nan * np.ones((self._size, obs_dim), dtype=np.float32)
        self._actions = np.nan * np.ones((self._size, action_dim), dtype=np.float32)
        self._rewards = np.nan * np.ones((self._size,), dtype=np.float32)
        self._dones = np.empty((self._size,), dtype=bool)
        self._index = 0
        self._curr_size = 0

        ### keep track of valid start indices (depends on horizon H)
        self._start_indices = np.zeros((self._size,), dtype=np.int16)

        ### keep track of statistics
        self._stats = defaultdict(int)

        ### logging
        self._last_done_index = 0
        self._log_stats = defaultdict(list)
        self._log_paths = []

    def __len__(self):
        return self._curr_size

    def _get_indices(self, start, end):
        if start < end:
            return list(range(start, end))
        elif start > end:
            return list(range(start, len(self))) + list(range(end))
        else:
            raise Exception

    ##################
    ### Statistics ###
    ##################

    @property
    def statistics(self):
        stats = dict()
        for name, value in (('observations', self._observations[:len(self)]),
                            ('actions', self._actions[:len(self)]),
                            ('rewards', self._rewards[:len(self)])):
            stats[name + '_mean'] = np.mean(value, axis=0)
            if np.shape(self._stats[name + '_mean']) is tuple():
                stats[name + '_mean'] = np.array([stats[name + '_mean']])
            stats[name + '_cov'] = np.cov(value.T)
            if np.shape(stats[name + '_cov']) is tuple():
                stats[name + '_cov'] = np.array([[stats[name + '_cov']]])
            orth, eigs, _ = np.linalg.svd(stats[name + '_cov'])
            stats[name + '_orth'] = orth / np.sqrt(eigs + 1e-5)

        return stats

    @staticmethod
    def statistics_pools(replay_pools):
        pool_stats = [replay_pool.statistics for replay_pool in replay_pools]
        pool_lens = np.array([len(replay_pool) for replay_pool in replay_pools]).astype(float)
        pool_ratios = pool_lens / pool_lens.sum()

        stats = defaultdict(int)
        for ratio, pool_stat in zip(pool_ratios, pool_stats):
            for k, v in [(k, v) for (k, v) in pool_stat.items() if 'mean' in k or 'cov' in k]:
                stats[k] += ratio * v
        for name in ('observations', 'actions', 'rewards'):
            orth, eigs, _ = np.linalg.svd(stats[name+'_cov'])
            stats[name+'_orth'] = orth / np.sqrt(eigs + 1e-5)

        return stats

    ###################
    ### Add to pool ###
    ###################

    def add(self, step, observation, action, reward, done):
        self._steps[self._index] = step
        self._observations[self._index, :] = self._env_spec.observation_space.flatten(observation)
        self._actions[self._index, :] = self._env_spec.action_space.flatten(action)
        self._rewards[self._index] = reward
        self._dones[self._index] = done
        self._index = (self._index + 1) % self._size
        self._curr_size = max(self._curr_size, self._index)

        if len(self) >= self._H:
            # unnulify H ago
            self._start_indices[self._index - self._H] = True
            # nullify next H, since may be from a different sequence
            false_indices = self._get_indices(self._index, self._index + self._H)
            self._start_indices[false_indices] = False

        ### update log stats
        self._update_log_stats(observation, action, reward, done)

    ########################
    ### Sample from pool ###
    ########################

    def can_sample(self):
        return len(self) > self._H

    def sample(self, batch_size):
        """
        :return observations, actions, and rewards of horizon H+1
        """
        if not self.can_sample():
            return None

        observations, actions, rewards, dones = [], [], [], []

        start_indices = np.random.choice(self._start_indices.nonzero()[0], size=(batch_size,), replace=True)
        for start_index in start_indices:
            indices = self._get_indices(start_index, (start_index + self._H + 1) % self._curr_size)
            observations_i = self._observations[indices]
            actions_i = self._actions[indices]
            rewards_i = self._rewards[indices]
            dones_i = self._dones[indices]
            if np.any(dones_i[:-1]):
                # H = 3
                # observations = [0 1 2 3]
                # actions = [10 11 12 13]
                # rewards = [20 21 22 23]
                # dones = [False True False False]

                d_idx = np.argmax(dones_i)
                for j in range(d_idx + 1, len(dones_i)):
                    observations_i[j, :] = 0.
                    actions_i[j, :] = self._env_spec.action_space.flatten(self._env_spec.action_space.sample())
                    rewards_i[j] = 0.
                    dones_i[j] = True

            if not np.all(np.isfinite(observations_i)):
                print('fuck null')
                import IPython; IPython.embed()

                # observations = [0 1 2 3]
                # actions = [10 11 rand rand]
                # rewards = [20 21 0 0]
                # dones = [False True True True]

            observations.append(observations_i.tolist())
            actions.append(actions_i.tolist())
            rewards.append(rewards_i.tolist())
            dones.append(dones_i.tolist())


        observations = np.asarray(observations)
        actions = np.asarray(actions)
        rewards = np.asarray(rewards)
        dones = np.asarray(dones)

        for k, arr in enumerate((observations, actions, rewards, dones)):
            if not np.all(np.isfinite(arr)):
                print('fuck fuck null')
                import IPython; IPython.embed()

            assert(np.all(np.isfinite(arr)))
            assert(arr.shape[0] == batch_size)
            assert(arr.shape[1] == self._H + 1)

        return observations, actions, rewards, dones

    @staticmethod
    def sample_pools(replay_pools, batch_size):
        """ Sample from replay pools (treating them as one big replay pool) """
        if not np.any([replay_pool.can_sample() for replay_pool in replay_pools]):
            return None

        observations, actions, rewards, dones = [], [], [], []

        # calculate ratio of pool sizes
        pool_lens = np.array([replay_pool.can_sample() * len(replay_pool) for replay_pool in replay_pools]).astype(float)
        pool_ratios = pool_lens / pool_lens.sum()
        # how many from each pool
        batch_sizes = np.zeros(len(replay_pools), dtype=int)
        for _ in range(batch_size):
            batch_sizes[np.random.choice(range(len(replay_pools)), p=pool_ratios)] += 1
        # sample from each pool
        for i, replay_pool in enumerate(replay_pools):
            if batch_sizes[i] == 0:
                continue

            observations_i, actions_i, rewards_i, dones_i = replay_pool.sample(batch_sizes[i])
            observations.append(observations_i)
            actions.append(actions_i)
            rewards.append(rewards_i)
            dones.append(dones_i)

        observations = np.vstack(observations)
        actions = np.vstack(actions)
        rewards = np.vstack(rewards)
        dones = np.vstack(dones)

        for arr in (observations, actions, rewards, dones):
            assert(len(arr) == batch_size)

        return observations, actions, rewards, dones

    ###############
    ### Logging ###
    ###############

    def _update_log_stats(self, observation, action, reward, done):
        if done:
            indices = self._get_indices(self._last_done_index, self._index)
            ### update log
            rewards = self._rewards[indices]
            self._log_stats['FinalReward'].append(rewards[-1])
            self._log_stats['AvgReward'].append(np.mean(rewards))
            self._log_stats['CumReward'].append(np.sum(rewards))

            for k, v in self._log_stats.items():
                if len(v) > self._log_history_len:
                    self._log_stats[k] = v[1:]

            # steps = self._steps[indices]
            # import IPython; IPython.embed()

            # for s in self._steps[indices]:
            #     print(s)

            ## update paths
            self._log_paths.append({
                'steps': self._steps[indices],
                'observations': self._observations[indices],
                'actions': self._actions[indices],
                'rewards': self._rewards[indices],
                'dones': self._dones[indices],
            })

            self._last_done_index = self._index

    def get_record_tabular(self):
        return {
            'FinalRewardMean': np.mean(self._log_stats['FinalReward']),
            'FinalRewardStd': np.std(self._log_stats['FinalReward']),
            'AvgRewardMean': np.mean(self._log_stats['AvgReward']),
            'AvgRewardStd': np.std(self._log_stats['AvgReward']),
            'CumRewardMean': np.mean(self._log_stats['CumReward']),
            'CumRewardStd': np.std(self._log_stats['CumReward'])
        }

    def get_recent_paths(self):
        paths = self._log_paths
        self._log_paths = []
        return paths

    @staticmethod
    def log_pools(replay_pools):
        record_tabulars = [replay_pool.get_record_tabular() for replay_pool in replay_pools]
        for key in sorted(record_tabulars[0].keys()):
            logger.record_tabular(key, np.mean([rc[key] for rc in record_tabulars]))

    @staticmethod
    def get_recent_paths_pools(replay_pools):
        return list(itertools.chain(*[rp.get_recent_paths() for rp in replay_pools]))
