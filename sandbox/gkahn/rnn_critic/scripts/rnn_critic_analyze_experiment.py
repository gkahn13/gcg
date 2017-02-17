import argparse
import os
import joblib
import itertools

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils.extmath import cartesian

from rllab.envs.gym_env import GymEnv
from rllab.sampler.utils import rollout as rollout_policy

from sandbox.gkahn.rnn_critic.envs.point_env import PointEnv
from sandbox.gkahn.rnn_critic.algos.rnn_util import Rollout

class AnalyzeRNNCritic(object):
    def __init__(self, folder):
        self._folder = folder

    #############
    ### Files ###
    #############

    def _itr_file(self, itr):
        return os.path.join(self._folder, 'itr_{0:d}.pkl'.format(itr))

    @property
    def _analyze_img_file(self):
        return os.path.join(self._folder, 'analyze.png')

    def _analyze_rollout_img_file(self, itr, is_train):
        return os.path.join(self._folder, 'analyze_{0}_rollout_itr_{1:d}.png'.format('train' if is_train else 'eval', itr))

    def _analyze_policy_img_file(self, itr):
        return os.path.join(self._folder, 'analyze_policy_itr_{0:d}.png'.format(itr))

    ####################
    ### Data loading ###
    ####################

    def _load_itr_policy(self, itr):
        d = joblib.load(self._itr_file(itr))
        return d['policy']

    def _load_itr(self, itr):
        graph = tf.Graph()
        with graph.as_default():
            sess = tf.Session(graph=graph)
            with sess.as_default():
                d = joblib.load(self._itr_file(itr))
                train_log = d['train_log']
                rollouts = d['rollouts']
                env = d['env']
                d['policy'].terminate()

        return train_log, rollouts, env

    def _load_all_itrs(self):
        train_log_itrs = []
        train_rollouts_itrs = []
        env_itrs = []

        itr = 0
        while os.path.exists(self._itr_file(itr)):
            train_log, rollouts, env = self._load_itr(itr)
            train_log_itrs.append(train_log)
            train_rollouts_itrs.append(rollouts)
            env_itrs.append(env)

            itr += 1

        return train_log_itrs, train_rollouts_itrs, env_itrs

    def _eval_all_policies(self, env_itrs):
        rollouts_itrs = []

        itr = 0
        while os.path.exists(self._itr_file(itr)):
            graph = tf.Graph()
            with graph.as_default():
                sess = tf.Session(graph=graph)
                with sess.as_default():
                    env = env_itrs[itr]
                    policy = self._load_itr_policy(itr)

                    rollouts = []
                    for _ in range(50):
                        path = rollout_policy(env, policy, max_path_length=env.horizon)
                        rollout = Rollout()
                        for obs, action, reward in zip(path['observations'], path['actions'], path['rewards']):
                            rollout.add(obs, action, reward, False)
                        rollouts.append(rollout)

            rollouts_itrs.append(rollouts)
            itr += 1

        return rollouts_itrs

    ################
    ### Plotting ###
    ################

    def _plot_analyze(self, train_log_itrs, train_rollouts_itrs, eval_rollouts_itrs):
        max_itr = len(train_log_itrs)

        f, axes = plt.subplots(3, 1, figsize=(3 * max_itr, 7.5))
        f.tight_layout()

        ### plot training cost
        ax = axes[0]
        costs = list(itertools.chain(*[train_log['cost'] for train_log in train_log_itrs]))
        costs = np.convolve(costs, (1 / 10.) * np.ones(10), mode='valid')
        itrs = np.linspace(0, max_itr, len(costs))
        ax.plot(itrs, costs, 'k-')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Cost')

        def plot_reward(ax, rewards):
            color = 'k'
            bp = ax.boxplot(rewards,
                            positions=np.arange(max_itr),
                            widths=0.25)
            for key in ('boxes', 'medians', 'whiskers', 'fliers', 'caps'):
                plt.setp(bp[key], color=color)
            for line in bp['medians']:
                # get position data for median line
                x, y = line.get_xydata()[1]  # top of median line
                # overlay median value
                ax.text(x, y, '%.2f' % y,
                        horizontalalignment='left',
                        verticalalignment='center',
                        color='r')  # draw above, centered
            for line in bp['boxes']:
                x, y = line.get_xydata()[0]  # bottom of left line
                ax.text(x, y, '%.2f' % y,
                        horizontalalignment='right',  # centered
                        verticalalignment='center',
                        color='k', alpha=0.5)  # below
                x, y = line.get_xydata()[3]  # bottom of right line
                ax.text(x, y, '%.2f' % y,
                        horizontalalignment='right',  # centered
                        verticalalignment='center',
                        color='k', alpha=0.5)  # below
            plt.setp(bp['fliers'], marker='_')
            plt.setp(bp['fliers'], markeredgecolor=color)
            ax.set_xlabel('Iteration')

        ### plot train final reward
        ax = axes[1]
        rewards = [[rollout.rewards[-1] for rollout in rollouts] for rollouts in train_rollouts_itrs]
        plot_reward(ax, rewards)
        ax.set_ylabel('Train final reward')

        ### plot eval final reward
        ax = axes[2]
        rewards = [[rollout.rewards[-1] for rollout in rollouts] for rollouts in eval_rollouts_itrs]
        plot_reward(ax, rewards)
        ax.set_ylabel('Eval final reward')

        f.savefig(self._analyze_img_file, bbox_inches='tight')
        plt.close(f)

    def _plot_rollouts(self, train_rollouts_itrs, eval_rollouts_itrs, env_itrs, is_train):
        env = env_itrs[0]
        while hasattr(env, 'wrapped_env'):
            env = env.wrapped_env
        if type(env) == PointEnv:
            self._plot_rollouts_PointEnv(train_rollouts_itrs, eval_rollouts_itrs, env_itrs, is_train)
        elif type(env) == GymEnv:
            if 'Reacher' in env.env_id:
                self._plot_rollouts_Reacher(train_rollouts_itrs, eval_rollouts_itrs, env_itrs, is_train)
        else:
            pass

    def _plot_rollouts_PointEnv(self, train_rollouts_itrs, eval_rollouts_itrs, env_itrs, is_train):
        rollouts_itrs = train_rollouts_itrs if is_train else eval_rollouts_itrs

        for itr, rollouts in enumerate(rollouts_itrs):

            N_rollouts = 25
            rollouts = sorted(rollouts, key=lambda r: r.rewards[-1], reverse=True)
            if len(rollouts) > N_rollouts:
                rollouts = rollouts[::int(np.ceil(len(rollouts)) / float(N_rollouts))]

            nrows = ncols = int(np.ceil(np.sqrt(len(rollouts))))
            f, axes = plt.subplots(nrows, ncols, figsize=(10, 10))

            all_positions = np.vstack([np.array(rollout.observations) for rollout in rollouts])
            xlim = ylim = (all_positions.min(), all_positions.max())

            for ax, rollout in zip(axes.ravel(), sorted(rollouts, key=lambda r: r.rewards[-1], reverse=True)):
                # plot all prior rollouts
                for train_rollout in itertools.chain(*train_rollouts_itrs[:itr + 1]):
                    train_positions = np.array(train_rollout.observations)
                    ax.plot(train_positions[:, 0], train_positions[:, 1], color='b', marker='', linestyle='-',
                            alpha=0.2)

                # plot this rollout
                positions = np.array(rollout.observations)
                ax.plot(positions[:, 0], positions[:, 1], color='k', marker='o', linestyle='-', markersize=0.2)
                ax.plot([0], [0], color='r', marker='x', markersize=5.)
                ax.plot([positions[0, 0]], [positions[0, 1]], color='g', marker='o', markersize=5.)
                ax.plot([positions[-1, 0]], [positions[-1, 1]], color='y', marker='o', markersize=5.)

                ax.set_xlim(xlim)
                ax.set_ylim(ylim)
                ax.set_title('{0:.2f}'.format(rollout.rewards[-1]))

            f.tight_layout()

            f.savefig(self._analyze_rollout_img_file(itr, is_train), bbox_inches='tight', dpi=200.)
            plt.close(f)

    def _plot_rollouts_Reacher(self, train_rollouts_itrs, eval_rollouts_itrs, env_itrs, is_train):
        def get_rollout_positions(rollout):
            observations = np.array(rollout.observations)
            goal_pos = observations[0, 4:6]
            positions = observations[:, -3:-1] + goal_pos
            return positions, goal_pos

        rollouts_itrs = train_rollouts_itrs if is_train else eval_rollouts_itrs

        for itr, rollouts in enumerate(rollouts_itrs):

            N_rollouts = 25
            rollouts = sorted(rollouts, key=lambda r: r.rewards[-1], reverse=True)
            if len(rollouts) > N_rollouts:
                rollouts = rollouts[::int(np.ceil(len(rollouts)) / float(N_rollouts))]

            nrows = ncols = int(np.ceil(np.sqrt(len(rollouts))))
            f, axes = plt.subplots(nrows, ncols, figsize=(10, 10))
            xlim = ylim = (-0.25, 0.25)

            for ax, rollout in zip(axes.ravel(), sorted(rollouts, key=lambda r: r.rewards[-1], reverse=True)):
                # plot all prior rollouts
                for train_rollout in itertools.chain(*train_rollouts_itrs[:itr + 1]):
                    train_positions, _ = get_rollout_positions(train_rollout)
                    ax.plot(train_positions[:, 0], train_positions[:, 1], color='b', marker='', linestyle='-',
                            alpha=0.2)

                # plot this rollout
                positions, goal_pos = get_rollout_positions(rollout)
                ax.plot(positions[:, 0], positions[:, 1], color='k', marker='o', linestyle='-', markersize=0.2)
                ax.plot([goal_pos[0]], [goal_pos[1]], color='r', marker='x', markersize=5.)
                ax.plot([positions[0, 0]], [positions[0, 1]], color='g', marker='o', markersize=5.)
                ax.plot([positions[-1, 0]], [positions[-1, 1]], color='y', marker='o', markersize=5.)

                ax.set_xlim(xlim)
                ax.set_ylim(ylim)
                ax.set_title('{0:.2f}'.format(rollout.rewards[-1]))

            f.tight_layout()

            f.savefig(self._analyze_rollout_img_file(itr, is_train), bbox_inches='tight', dpi=200.)
            plt.close(f)

    def _plot_policies(self, rollouts_itrs, env_itrs):
        env = env_itrs[0].wrapped_env
        if type(env) == PointEnv:
            self._plot_policies_PointEnv(rollouts_itrs, env_itrs)
        else:
            pass

    def _plot_policies_PointEnv(self, rollouts_itrs, env_itrs):
        itr = 0
        while os.path.exists(self._itr_file(itr)):
            N = 5
            f, axes = plt.subplots(N, N, figsize=(10, 10))

            policy = self._load_itr_policy(itr)

            observations = cartesian([np.linspace(l, u, N) for l, u in zip([-1., -1.], [1., 1.])])
            for ax, observation in zip(np.fliplr(axes.T).ravel(), observations):
                action, _ = policy.get_action(observation)
                ax.arrow(observation[0], observation[1], action[0], action[1], head_width=0.1, color='k')
                ax.plot([0], [0], color='r', marker='x', markersize=3.)
                ax.set_xlim((-2, 2))
                ax.set_ylim((-2, 2))

            f.suptitle('Itr {0:d}'.format(itr))
            f.savefig(self._analyze_policy_img_file(itr), bbox_inches='tight', dpi=200.)
            plt.close(f)

            itr += 1
            policy.terminate()

    ###########
    ### Run ###
    ###########

    def run(self):
        train_log_itrs, train_rollouts_itrs, env_itrs = self._load_all_itrs()
        eval_rollouts_itrs = self._eval_all_policies(env_itrs)
        self._plot_analyze(train_log_itrs, train_rollouts_itrs, eval_rollouts_itrs)
        self._plot_rollouts(train_rollouts_itrs, eval_rollouts_itrs, env_itrs, is_train=False)
        self._plot_rollouts(train_rollouts_itrs, eval_rollouts_itrs, env_itrs, is_train=True)
        self._plot_policies(train_rollouts_itrs, env_itrs)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('folder', type=str)
    args = parser.parse_args()

    analyze = AnalyzeRNNCritic(os.path.join('/home/gkahn/code/rllab/data/local/rnn-critic/', args.folder))
    analyze.run()
