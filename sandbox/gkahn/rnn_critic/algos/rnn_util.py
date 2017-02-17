from collections import defaultdict
import numpy as np

class Rollout(object):
    def __init__(self):
        self.observations = []
        self.actions = []
        self.rewards = []
        self.terminals = []

    def add(self, observation, action, reward, terminal):
        self.observations.append(observation)
        self.actions.append(action)
        self.rewards.append(reward)
        self.terminals.append(terminal)

    def __getitem__(self, item):
        return self.observations[item], self.actions[item], self.rewards[item], self.terminals[item]

    def __len__(self):
        return len(self.observations)

class RNNCriticReplayPool(object):
    def __init__(self, max_pool_size=1e6):
        self._max_pool_size = max_pool_size
        self._rollouts = []

        self._means = defaultdict(int)
        self._covs = defaultdict(int)
        self._orths = dict()
        self._N_steps = 0

    def add_rollout(self, rollout):
        """
        :type rollout: Rollout
        """
        self._rollouts.append(rollout)

        # update mean/orth
        new_N_steps = self._N_steps + len(rollout)
        for name, new_values in (('observation', rollout.observations),
                                 ('action', rollout.actions),
                                 ('reward', rollout.rewards)):
            self._means[name] = (self._N_steps / float(new_N_steps)) * self._means[name] + \
                                (len(rollout) / float(new_N_steps)) * np.mean(new_values, axis=0)
            if np.shape(self._means[name]) is tuple():
                self._means[name] = np.array([self._means[name]])
            self._covs[name] = (self._N_steps / float(new_N_steps)) * self._covs[name] + \
                               (len(rollout) / float(new_N_steps)) * np.cov(np.transpose(new_values))
            if np.shape(self._covs[name]) is tuple():
                self._covs[name] = np.array([[self._covs[name]]])
            orth, eigs, _ = np.linalg.svd(self._covs[name])
            self._orths[name] = orth / np.sqrt(eigs + 1e-5)

        self._N_steps = new_N_steps

        # make sure pool size is under max pool size
        while sum([len(r) for r in self._rollouts]) > self._max_pool_size:
            self._rollouts = self._rollouts[1:]

    def get_rollouts(self, last_n_rollouts=None):
        if last_n_rollouts is None:
            last_n_rollouts = len(self._rollouts)

        return self._rollouts[-last_n_rollouts:]

    def get_random_sequence(self, H):
        """
        :param H: length of random contiguous sequence
        :return observations, actions, rewards, terminals
        """
        # sample rollout according to relative length
        rollouts = self.get_rollouts()
        rollout_lens = [(len(rollout) >= H) * len(rollout) for rollout in rollouts]
        rollout_ratios = np.asarray(rollout_lens, dtype=float) / sum(rollout_lens)
        rollout_idx = np.random.choice(range(len(rollouts)), p=rollout_ratios)
        rollout = rollouts[rollout_idx]
        # rollout = np.random.choice(self._rollouts, p=rollout_ratios)

        # for the chosen rollout, sample a contiguous sequence of length H
        start = np.random.randint(0, len(rollout) - H)
        return rollout[slice(start, start+H)]

    def get_whitening(self):
        return self._means['observation'], self._orths['observation'], \
               self._means['action'], self._orths['action'], \
               self._means['reward'], self._orths['reward']

    def statistics(self, last_n_rollouts=None):
        rollouts = self.get_rollouts(last_n_rollouts=last_n_rollouts)
        return {
            'final_reward': [r.rewards[-1] for r in rollouts],
            'avg_reward': [np.mean(r.rewards) for r in rollouts]
        }

    def __len__(self):
        return len(self._rollouts)
