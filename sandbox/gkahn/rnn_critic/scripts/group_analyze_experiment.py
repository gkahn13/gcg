import os
import itertools
import functools

import numpy as np
import scipy
import matplotlib.pyplot as plt
from matplotlib import ticker

from sandbox.rocky.tf.envs.base import TfEnv
from rllab.envs.normalized_env import normalize

from sandbox.gkahn.rnn_critic.envs.chain_env import ChainEnv
from sandbox.gkahn.rnn_critic.envs.sparse_point_env import SparsePointEnv
from sandbox.gkahn.rnn_critic.envs.point_env import PointEnv
from sandbox.gkahn.rnn_critic.envs.phd_env import PhdEnv
from sandbox.gkahn.rnn_critic.envs.cartpole_swingup_env import CartPoleSwingupEnv
from rllab.envs.gym_env import GymEnv

from sandbox.gkahn.rnn_critic.examples.analyze_experiment import AnalyzeRNNCritic

### environments
import gym
from sandbox.gkahn.rnn_critic.envs.atari_wrappers import wrap_deepmind
from sandbox.gkahn.rnn_critic.envs.pygame_wrappers import wrap_pygame
from rllab.envs.gym_env import GymEnv
from sandbox.gkahn.rnn_critic.envs.premade_gym_env import PremadeGymEnv
import gym_ple
from sandbox.gkahn.rnn_critic.envs.point_env import PointEnv
from sandbox.gkahn.rnn_critic.envs.sparse_point_env import SparsePointEnv
from sandbox.gkahn.rnn_critic.envs.chain_env import ChainEnv

##################
### Processing ###
##################

def flatten_list(l):
    return [val for sublist in l for val in sublist]

class DataAverageInterpolation(object):
    def __init__(self):
        self.xs = []
        self.ys = []
        self.fs = []

    def add_data(self, x, y):
        self.xs.append(x)
        self.ys.append(y)
        self.fs.append(scipy.interpolate.interp1d(x, y))

    def eval(self, x):
        ys = [f(x) for f in self.fs]
        return np.array(np.mean(ys, axis=0)), np.array(np.std(ys, axis=0))

def create_best_fit_axes(num_axes, **kwargs):
    def calculate_factors(n):
        return sorted(list(functools.reduce(list.__add__, ([i, n // i] for i in range(1, int(n ** 0.5) + 1) if n % i == 0))))

    n = num_axes
    while True:
        factors = calculate_factors(n)
        if len([f for f in factors if f >=4 and f <= n//4]) > 0:
            break
        n += 1

    rows = factors[len(factors)//2]
    columns = n // rows

    return plt.subplots(rows, columns, **kwargs)

################
### Analysis ###
################

class PlotAnalyzeRNNCritic(object):
    def __init__(self, save_folder, name, analyze_groups):
        """
        :param analyze_groups: [[AnalyzeRNNCritic, AnalyzeRNNCritic, ...], [AnalyzeRNNCritic, AnalyzeRNNCritic, ...]]
        """
        self._save_folder = save_folder
        self._name = name
        self._analyze_groups = analyze_groups

        env = analyze_groups[0][0].env
        if env is None:
            inner_env = eval(analyze_groups[0][0].params['alg']['env'])
            env = TfEnv(normalize(inner_env))
        while hasattr(env, 'wrapped_env'):
            env = env.wrapped_env
        self._env = env

    #############
    ### Files ###
    #############

    @property
    def _folder(self):
        folder = os.path.join(self._save_folder, self._name)
        if not os.path.exists(folder):
            os.mkdir(folder)
        return folder

    @property
    def _analyze_img_file(self):
        return os.path.join(self._folder, '{0}_analyze.png'.format(self._name))

    ################
    ### Plotting ###
    ################

    def _plot_analyze(self):
        if isinstance(self._env, ChainEnv):
            self._plot_analyze_ChainEnv()
        elif isinstance(self._env, PointEnv):
            self._plot_analyze_PointEnv()
        elif isinstance(self._env, SparsePointEnv):
            self._plot_analyze_SparsePointEnv()
        elif isinstance(self._env, PhdEnv):
            self._plot_analyze_PhdEnv()
        elif isinstance(self._env, GymEnv) and 'CartPole' in self._env.env_id:
            self._plot_analyze_CartPole()
        elif isinstance(self._env, GymEnv) and 'catcher' in self._env.env_id.lower():
            self._plot_analyze_Catcher()
        elif isinstance(self._env, GymEnv) and 'pong' in self._env.env_id.lower():
            self._plot_analyze_Pong()
        elif isinstance(self._env, GymEnv) and 'invertedpendulum' in self._env.env_id.lower():
            if 'offpolicy' in analyze_groups[0][0].params['alg']:
                self._plot_analyze_InvertedPendulum_offpolicy()
            else:
                self._plot_analyze_InvertedPendulum()
        elif isinstance(self._env, GymEnv) and 'tennis' in self._env.env_id.lower():
            self._plot_analyze_Tennis()
        elif isinstance(self._env, GymEnv) and 'reacher' in self._env.env_id.lower():
            self._plot_analyze_Reacher()
        elif isinstance(self._env, GymEnv) and 'cheetah' in self._env.env_id.lower():
            self._plot_analyze_HalfCheetah()
        elif isinstance(self._env, GymEnv) and 'pendulum' in self._env.env_id.lower():
            self._plot_analyze_Pendulum()
        elif isinstance(self._env, GymEnv) and 'hopper' in self._env.env_id.lower():
            self._plot_analyze_Hopper()
        elif isinstance(self._env, CartPoleSwingupEnv):
            self._plot_analyze_CartPoleSwingup()
        else:
            pass

    def _plot_analyze_ChainEnv(self):
        f, axes = plt.subplots(1+len(self._analyze_groups), 1, figsize=(15, 5*len(self._analyze_groups)), sharex=True)

        ### plot training cost
        ax = axes[0]
        for analyze_group in self._analyze_groups:
            data_interp = DataAverageInterpolation()
            for analyze in analyze_group:
                steps = analyze.progress['Step'][1:]
                costs = analyze.progress['Cost'][1:]
                data_interp.add_data(steps, costs)

            steps = np.r_[min(flatten_list(data_interp.xs)):max(flatten_list(data_interp.xs)):0.01]
            costs_mean, costs_std = data_interp.eval(steps)

            ax.plot(steps, costs_mean, color=analyze.plot['color'], label=analyze.plot['label'])
            ax.fill_between(steps, costs_mean - costs_std, costs_mean + costs_std,
                            color=analyze.plot['color'], alpha=0.4)
        ax.set_ylabel('Training cost')
        ax.legend(loc='upper right')

        ### plot training rollout length vs step
        for ax, analyze_group in zip(axes[1:], self._analyze_groups):
            data_interp = DataAverageInterpolation()
            min_step = max_step = None
            for analyze in analyze_group:
                rollouts = list(itertools.chain(*analyze.train_rollouts_itrs))
                rollout_lens = [len(r['observations']) for r in rollouts]
                steps = [r['steps'][-1] for r in rollouts]
                def moving_avg_std(idxs, data, window):
                    means, stds = [], []
                    for i in range(window, len(data)):
                        means.append(np.mean(data[i - window:i]))
                        stds.append(np.std(data[i - window:i]))
                    return idxs[window:], np.asarray(means), np.asarray(stds)
                moving_steps, moving_rollout_lens, _ = moving_avg_std(steps, rollout_lens, 5)
                data_interp.add_data(moving_steps, moving_rollout_lens)
                if min_step is None:
                    min_step = moving_steps[0]
                if max_step is None:
                    max_step = moving_steps[-1]
                min_step = max(min_step, moving_steps[0])
                max_step = min(max_step, moving_steps[-1])
            steps = np.r_[min_step:max_step:0.01]
            lens_mean, lens_std = data_interp.eval(steps)

            ax.plot(steps, lens_mean, color=analyze.plot['color'], label=analyze.plot['label'])
            ax.fill_between(steps, lens_mean - lens_std, lens_mean + lens_std,
                            color=analyze.plot['color'], alpha=0.4)
            ax.vlines(analyze.params['alg']['learn_after_n_steps'], 0, ax.get_ylim()[1], colors='k', linestyles='dashed')
            ax.hlines(analyze.env.spec.observation_space.n, steps[0], steps[-1], colors='k', linestyles='dashed')
            ax.set_ylabel('Rollout length')

        ### for all plots
        ax.set_xlabel('Steps')
        xfmt = ticker.ScalarFormatter()
        xfmt.set_powerlimits((0, 0))
        ax.xaxis.set_major_formatter(xfmt)

        f.savefig(self._analyze_img_file, bbox_inches='tight')
        plt.close(f)

    def _plot_analyze_PointEnv(self):
        f, axes = plt.subplots(1 + len(self._analyze_groups), 1, figsize=(15, 5 * len(self._analyze_groups)),
                               sharex=True)

        ### plot training cost
        ax = axes[0]
        for analyze_group in self._analyze_groups:
            data_interp = DataAverageInterpolation()
            for analyze in analyze_group:
                steps = analyze.progress['Step'][1:]
                costs = analyze.progress['Cost'][1:]
                data_interp.add_data(steps, costs)

            steps = np.r_[np.min(np.hstack(data_interp.xs)):np.max(np.hstack(data_interp.xs)):0.01]
            costs_mean, costs_std = data_interp.eval(steps)

            ax.plot(steps, costs_mean, color=analyze.plot['color'], label=analyze.plot['label'])
            ax.fill_between(steps, costs_mean - costs_std, costs_mean + costs_std,
                            color=analyze.plot['color'], alpha=0.4)
        ax.set_ylabel('Training cost')
        ax.legend(loc='upper right')

        ### plot training rollout final reward vs step
        for ax, analyze_group in zip(axes[1:], self._analyze_groups):
            data_interp = DataAverageInterpolation()
            min_step = max_step = None
            for analyze in analyze_group:
                rollouts = list(itertools.chain(*analyze.train_rollouts_itrs))
                final_rewards = [r['rewards'][-1] for r in rollouts]
                steps = [r['steps'][-1] for r in rollouts]

                data_interp.add_data(steps, final_rewards)
                if min_step is None:
                    min_step = steps[0]
                if max_step is None:
                    max_step = steps[-1]
                min_step = max(min_step, steps[0])
                max_step = min(max_step, steps[-1])
            steps = np.r_[min_step:max_step:0.01]
            final_rewards_mean, final_rewards_std = data_interp.eval(steps)

            ax.plot(steps, final_rewards_mean, color=analyze.plot['color'], label=analyze.plot['label'])
            ax.fill_between(steps, final_rewards_mean - final_rewards_std, final_rewards_mean + final_rewards_std,
                            color=analyze.plot['color'], alpha=0.4)
            ax.vlines(analyze.params['alg']['learn_after_n_steps'], ax.get_ylim()[0], 0, colors='k',
                      linestyles='dashed')
            ax.hlines(0, steps[0], steps[-1], colors='k',
                      linestyles='dashed')
            for step in ax.get_xticks():
                step = np.clip(step, min_step, max_step)
                ax.text(step, 0, '%.2f'%data_interp.eval([step])[0][0],
                        horizontalalignment='center',
                        verticalalignment='bottom',
                        color='k', alpha=0.8)
            ax.set_ylabel('Final reward')

        ### for all plots
        ax.set_xlabel('Steps')
        xfmt = ticker.ScalarFormatter()
        xfmt.set_powerlimits((0, 0))
        ax.xaxis.set_major_formatter(xfmt)

        y_min = min([ax.get_ylim()[0] for ax in axes[1:].ravel()])
        for ax in axes[1:].ravel():
            ax.set_ylim((y_min, 1.))

        f.savefig(self._analyze_img_file, bbox_inches='tight')
        plt.close(f)

    def _plot_analyze_SparsePointEnv(self):

        ##############################
        ### Plot fraction one axis ###
        ##############################

        f, axes = plt.subplots(1, 2, figsize=(30, 10))

        ### plot training cost
        ax = axes.ravel()[0]
        for analyze_group in self._analyze_groups:
            data_interp = DataAverageInterpolation()
            for analyze in analyze_group:
                steps = analyze.progress['Step'][1:]
                costs = analyze.progress['Cost'][1:]
                data_interp.add_data(steps, costs)

            steps = np.r_[np.min(np.hstack(data_interp.xs)):np.max(np.hstack(data_interp.xs)):0.01]
            costs_mean, costs_std = data_interp.eval(steps)

            ax.plot(steps, costs_mean, color=analyze.plot['color'], label=analyze.plot['label'])
            ax.fill_between(steps, costs_mean - costs_std, costs_mean + costs_std,
                            color=analyze.plot['color'], alpha=0.4)
        ax.set_ylabel('Training cost')
        ax.legend(loc='upper right')

        ### plot number of rollouts within radius (cdf) versus time step
        ax = axes.ravel()[1]
        radius = float(analyze_groups[0][0].params['alg']['env'].split('(')[-1].split(')')[0])
        for analyze_group in self._analyze_groups:
            data_interp = DataAverageInterpolation()
            min_step = max_step = None
            for analyze in analyze_group:
                rollouts = list(itertools.chain(*analyze.train_rollouts_itrs))
                rollouts = sorted(rollouts, key=lambda r: r['steps'][-1])
                steps, insides = [], []
                for r in rollouts:
                    steps.append(r['steps'][-1])
                    insides.append(float(np.linalg.norm(r['observations'][-1]) < radius))

                def moving_avg_std(idxs, data, window):
                    avg_idxs, means, stds = [], [], []
                    for i in range(window, len(data)):
                        avg_idxs.append(np.mean(idxs[i - window:i]))
                        means.append(np.mean(data[i - window:i]))
                        stds.append(np.std(data[i - window:i]))
                    return avg_idxs, np.asarray(means), np.asarray(stds)

                steps, insides, _ = moving_avg_std(steps, insides, window=10)

                data_interp.add_data(steps, insides)
                if min_step is None:
                    min_step = steps[0]
                if max_step is None:
                    max_step = steps[-1]
                min_step = max(min_step, steps[0])
                max_step = min(max_step, steps[-1])

            steps = np.r_[min_step:max_step:5.]
            insides_mean, insides_std = data_interp.eval(steps)

            ax.plot(steps, insides_mean, color=analyze.plot['color'], label=analyze.plot['label'])
            ax.grid()
        ax.set_ylabel('Fraction of rollouts within radius {0:.2f}'.format(radius))

        ### for all plots
        for ax in axes.ravel():
            ax.set_xlabel('Steps')
            xfmt = ticker.ScalarFormatter()
            xfmt.set_powerlimits((0, 0))
            ax.xaxis.set_major_formatter(xfmt)

        f.savefig(self._analyze_img_file, bbox_inches='tight', dpi=200)
        plt.close(f)

        ###################################
        ### Plot fraction multiple axes ###
        ###################################

        # f, axes = plt.subplots(1 + len(self._analyze_groups), 1, figsize=(15, 5 * len(self._analyze_groups)),
        #                        sharex=True)
        #
        # ### plot training cost
        # ax = axes.ravel()[0]
        # for analyze_group in self._analyze_groups:
        #     data_interp = DataAverageInterpolation()
        #     for analyze in analyze_group:
        #         steps = analyze.progress['Step'][1:]
        #         costs = analyze.progress['Cost'][1:]
        #         data_interp.add_data(steps, costs)
        #
        #     steps = np.r_[np.min(np.hstack(data_interp.xs)):np.max(np.hstack(data_interp.xs)):0.01]
        #     costs_mean, costs_std = data_interp.eval(steps)
        #
        #     ax.plot(steps, costs_mean, color=analyze.plot['color'], label=analyze.plot['label'])
        #     ax.fill_between(steps, costs_mean - costs_std, costs_mean + costs_std,
        #                     color=analyze.plot['color'], alpha=0.4)
        # ax.set_ylabel('Training cost')
        # ax.legend(loc='upper right')
        #
        # ### plot fraction rollouts within radius vs step
        # radius = float(analyze_groups[0][0].params['alg']['env'].split('(')[-1].split(')')[0])
        # for ax, analyze_group in zip(axes.ravel()[1:], self._analyze_groups):
        #     data_interp = DataAverageInterpolation()
        #     min_step = max_step = None
        #     for analyze in analyze_group:
        #         rollouts = list(itertools.chain(*analyze.train_rollouts_itrs))
        #         rollouts = sorted(rollouts, key=lambda r: r['steps'][-1])
        #         steps, insides = [], []
        #         for r in rollouts:
        #             steps.append(r['steps'][-1])
        #             insides.append(float(np.linalg.norm(r['observations'][-1]) < radius))
        #
        #         def moving_avg_std(idxs, data, window):
        #             means, stds = [], []
        #             for i in range(window, len(data)):
        #                 means.append(np.mean(data[i - window:i]))
        #                 stds.append(np.std(data[i - window:i]))
        #             return idxs[window:], np.asarray(means), np.asarray(stds)
        #
        #         steps, insides, _ = moving_avg_std(steps, insides, window=5)
        #
        #         data_interp.add_data(steps, insides)
        #         if min_step is None:
        #             min_step = steps[0]
        #         if max_step is None:
        #             max_step = steps[-1]
        #         min_step = max(min_step, steps[0])
        #         max_step = min(max_step, steps[-1])
        #
        #     steps = np.r_[min_step:max_step:5.]
        #     insides_mean, insides_std = data_interp.eval(steps)
        #
        #     ax.plot(steps, insides_mean, color=analyze.plot['color'], label=analyze.plot['label'])
        #     ax.fill_between(steps, insides_mean - insides_std, insides_mean + insides_std,
        #                     color=analyze.plot['color'], alpha=0.4)
        #     ax.grid()
        #
        # ### for all plots
        # ax.set_xlabel('Steps')
        # xfmt = ticker.ScalarFormatter()
        # xfmt.set_powerlimits((0, 0))
        # ax.xaxis.set_major_formatter(xfmt)
        #
        # f.savefig(self._analyze_img_file, bbox_inches='tight')
        # plt.close(f)


        ################
        ### Plot CDF ###
        ################

        # f, axes = plt.subplots(1, 2, figsize=(10, 5))
        #
        # ### plot training cost
        # ax = axes.ravel()[0]
        # for analyze_group in self._analyze_groups:
        #     data_interp = DataAverageInterpolation()
        #     for analyze in analyze_group:
        #         steps = analyze.progress['Step'][1:]
        #         costs = analyze.progress['Cost'][1:]
        #         data_interp.add_data(steps, costs)
        #
        #     steps = np.r_[np.min(np.hstack(data_interp.xs)):np.max(np.hstack(data_interp.xs)):0.01]
        #     costs_mean, costs_std = data_interp.eval(steps)
        #
        #     ax.plot(steps, costs_mean, color=analyze.plot['color'], label=analyze.plot['label'])
        #     ax.fill_between(steps, costs_mean - costs_std, costs_mean + costs_std,
        #                     color=analyze.plot['color'], alpha=0.4)
        # ax.set_ylabel('Training cost')
        # ax.legend(loc='upper right')
        #
        # ### plot number of rollouts within radius (cdf) versus time step
        # ax = axes.ravel()[1]
        # radius = float(analyze_groups[0][0].params['alg']['env'].split('(')[-1].split(')')[0])
        # for analyze_group in self._analyze_groups:
        #
        #     inside_timesteps = [] # final timestep for rollouts ending inside the radius
        #     for analyze in analyze_group:
        #         rollouts = list(itertools.chain(*analyze.train_rollouts_itrs))
        #         for r in rollouts:
        #             if np.linalg.norm(r['observations'][-1]) < radius:
        #                 inside_timesteps.append(r['steps'][-1])
        #     inside_timesteps = sorted(np.array(inside_timesteps, dtype=float) / float(len(analyze_group)))
        #     ax.plot([0] + inside_timesteps, np.linspace(0, len(inside_timesteps), len(inside_timesteps) + 1),
        #             color=analyze.plot['color'], label=analyze.plot['label'])
        # ax.set_ylabel('Rollouts within radius {0:.2f}'.format(radius))
        #
        # for ax in axes.ravel():
        #     ax.set_xlabel('Steps')
        #     ax.set_xlabel('Steps')
        #     xfmt = ticker.ScalarFormatter()
        #     xfmt.set_powerlimits((0, 0))
        #     ax.xaxis.set_major_formatter(xfmt)
        #
        # f.savefig(self._analyze_img_file, bbox_inches='tight', dpi=200)
        # plt.close(f)

    def _plot_analyze_PhdEnv(self):
        import IPython; IPython.embed()

        # f, axes = create_best_fit_axes(len(self._analyze_groups), figsize=(15, 15), sharex=True)
        f, axes = plt.subplots(len(self._analyze_groups), 1, figsize=(15, 15), sharex=True, sharey=True)

        ### plot episode length versus time step
        for ax, analyze_group in zip(axes.ravel(), self._analyze_groups):
            data_interp = DataAverageInterpolation()
            min_step = max_step = None
            for analyze in analyze_group:
                rollouts = list(itertools.chain(*analyze.eval_rollouts_itrs))
                steps = [r['steps'][0] for r in rollouts]
                eplens = [len(r['rewards']) for r in rollouts]

                def moving_avg_std(idxs, data, window):
                    avg_idxs, means, stds = [], [], []
                    for i in range(window, len(data)):
                        avg_idxs.append(np.mean(idxs[i - window:i]))
                        means.append(np.mean(data[i - window:i]))
                        stds.append(np.std(data[i - window:i]))
                    return avg_idxs, np.asarray(means), np.asarray(stds)

                steps, eplens, _ = moving_avg_std(steps, eplens, window=50)

                data_interp.add_data(steps, eplens)
                if min_step is None:
                    min_step = steps[0]
                if max_step is None:
                    max_step = steps[-1]
                min_step = max(min_step, steps[0])
                max_step = min(max_step, steps[-1])

            steps = np.r_[min_step:max_step:1000.][1:-1]
            eplens_mean, eplens_std = data_interp.eval(steps)

            ax.plot(steps, eplens_mean, color=analyze.plot['color'], label=analyze.plot['label'])
            ax.fill_between(steps, eplens_mean - eplens_std, eplens_mean + eplens_std,
                            color=analyze.plot['color'], alpha=0.4)

            ax.grid()
            ax.set_title(analyze_group[0].plot['label'], {'fontsize': 10})
            ax.set_ylabel('Episode length')
            ax.set_xlabel('Steps')
            xfmt = ticker.ScalarFormatter()
            xfmt.set_powerlimits((0, 0))
            ax.xaxis.set_major_formatter(xfmt)

            length = int(analyze.params['alg']['env'].split('length=')[-1].split(',')[0])
            ax.hlines(length + 1, steps[0], steps[-1], color='g', linestyle='--')
            ax.set_ylim((-1, 1.1*(length+1)))

        ### for all the plots
        for ax in axes.ravel():
            ax.vlines(analyze.params['alg']['learn_after_n_steps'], ax.get_ylim()[0], ax.get_ylim()[1],
                      color='k', linestyle='--')

        plt.tight_layout()
        f.savefig(self._analyze_img_file, bbox_inches='tight', dpi=200)
        plt.close(f)

    def _plot_analyze_CartPole(self):
        ##############################
        ### Plot fraction one axis ###
        ##############################

        f, axes = plt.subplots(1 + len(self._analyze_groups), 1, figsize=(15, 5 * len(self._analyze_groups)), sharex=True)

        ### plot training cost
        ax = axes.ravel()[0]
        for analyze_group in self._analyze_groups:
            data_interp = DataAverageInterpolation()
            min_step = max_step = None
            for analyze in analyze_group:
                steps = np.array(analyze.progress['Step'])
                costs = np.array(analyze.progress['Cost'])
                data_interp.add_data(steps, costs)

                if min_step is None:
                    min_step = steps[0]
                if max_step is None:
                    max_step = steps[-1]
                min_step = max(min_step, steps[0])
                max_step = min(max_step, steps[-1])

            steps = np.r_[min_step:max_step:1.]
            costs_mean, costs_std = data_interp.eval(steps)

            ax.plot(steps, costs_mean, color=analyze.plot['color'], label=analyze.plot['label'])
            ax.fill_between(steps, costs_mean - costs_std, costs_mean + costs_std,
                            color=analyze.plot['color'], alpha=0.4)
        ax.set_ylabel('Training cost')
        ax.legend(loc='upper right')

        ### plot cum reward versus time step
        for ax, analyze_group in zip(axes.ravel()[1:], self._analyze_groups):
            data_interp = DataAverageInterpolation()
            min_step = max_step = None
            for analyze in analyze_group:
                rollouts = list(itertools.chain(*analyze.train_rollouts_itrs))
                rollouts = sorted(rollouts, key=lambda r: r['steps'][-1])
                steps, cum_rewards = [], []
                for r in rollouts:
                    steps.append(r['steps'][-1])
                    cum_rewards.append(np.sum(r['rewards']))

                def moving_avg_std(idxs, data, window):
                    avg_idxs, means, stds = [], [], []
                    for i in range(window, len(data)):
                        avg_idxs.append(np.mean(idxs[i - window:i]))
                        means.append(np.mean(data[i - window:i]))
                        stds.append(np.std(data[i - window:i]))
                    return avg_idxs, np.asarray(means), np.asarray(stds)

                steps, cum_rewards, _ = moving_avg_std(steps, cum_rewards, window=10)

                data_interp.add_data(steps, cum_rewards)
                if min_step is None:
                    min_step = steps[0]
                if max_step is None:
                    max_step = steps[-1]
                min_step = max(min_step, steps[0])
                max_step = min(max_step, steps[-1])

            steps = np.r_[min_step:max_step:50.][1:-1]
            cum_rewards_mean, cum_rewards_std = data_interp.eval(steps)

            ax.plot(steps, cum_rewards_mean, color=analyze.plot['color'], label=analyze.plot['label'])
            ax.fill_between(steps, cum_rewards_mean - cum_rewards_std, cum_rewards_mean + cum_rewards_std,
                            color=analyze.plot['color'], alpha=0.4)
            ax.set_ylim((0, 225))
            ax.grid()
            ax.set_title(' '.join([analyze.name for analyze in analyze_group]))
        ax.set_ylabel('Cumulative reward')

        ### for all plots
        ax.set_xlabel('Steps')
        xfmt = ticker.ScalarFormatter()
        xfmt.set_powerlimits((0, 0))
        ax.xaxis.set_major_formatter(xfmt)

        f.savefig(self._analyze_img_file, bbox_inches='tight', dpi=200)
        plt.close(f)

    def _plot_analyze_Catcher(self):
        ##############################
        ### Plot fraction one axis ###
        ##############################

        f, axes = plt.subplots(1 + len(self._analyze_groups), 1, figsize=(15, 5 * len(self._analyze_groups)), sharex=True)

        ### plot training cost
        ax = axes.ravel()[0]
        for analyze_group in self._analyze_groups:
            data_interp = DataAverageInterpolation()
            min_step = max_step = None
            for analyze in analyze_group:
                steps = np.array(analyze.progress['Step'])
                costs = np.array(analyze.progress['Cost'])
                data_interp.add_data(steps, costs)

                if min_step is None:
                    min_step = steps[0]
                if max_step is None:
                    max_step = steps[-1]
                min_step = max(min_step, steps[0])
                max_step = min(max_step, steps[-1])

            steps = np.r_[min_step:max_step:100.]
            costs_mean, costs_std = data_interp.eval(steps)

            ax.plot(steps, costs_mean, color=analyze.plot['color'], label=analyze.plot['label'])
            ax.fill_between(steps, costs_mean - costs_std, costs_mean + costs_std,
                            color=analyze.plot['color'], alpha=0.4)
        ax.set_ylabel('Training cost')
        ax.legend(loc='upper right')

        ### plot fraction success vs time step
        for ax, analyze_group in zip(axes.ravel()[1:], self._analyze_groups):
            data_interp = DataAverageInterpolation()
            min_step = max_step = None
            for analyze in analyze_group:
                rollouts = list(itertools.chain(*analyze.train_rollouts_itrs))
                rollouts = sorted(rollouts, key=lambda r: r['steps'][-1])
                steps, successes = [], []
                for r in rollouts:
                    steps.append(r['steps'][-1])
                    successes.append(float(r['rewards'][-1] > 0))

                def moving_avg_std(idxs, data, window):
                    avg_idxs, means, stds = [], [], []
                    for i in range(window, len(data)):
                        avg_idxs.append(np.mean(idxs[i - window:i]))
                        means.append(np.mean(data[i - window:i]))
                        stds.append(np.std(data[i - window:i]))
                    return avg_idxs, np.asarray(means), np.asarray(stds)

                steps, successes, _ = moving_avg_std(steps, successes, window=100)

                data_interp.add_data(steps, successes)
                if min_step is None:
                    min_step = steps[0]
                if max_step is None:
                    max_step = steps[-1]
                min_step = max(min_step, steps[0])
                max_step = min(max_step, steps[-1])

            steps = np.r_[min_step:max_step:50.][1:-1]
            successes_mean, successes_std = data_interp.eval(steps)

            ax.plot(steps, successes_mean, color=analyze.plot['color'], label=analyze.plot['label'])
            ax.fill_between(steps, successes_mean - successes_std, successes_mean + successes_std,
                            color=analyze.plot['color'], alpha=0.4)
            ax.set_ylim((-0.1, 1.1))
            ax.grid()
            ax.set_title(' '.join([analyze.name for analyze in analyze_group]))
        ax.set_ylabel('Cumulative reward')

        ### for all plots
        ax.set_xlabel('Steps')
        xfmt = ticker.ScalarFormatter()
        xfmt.set_powerlimits((0, 0))
        ax.xaxis.set_major_formatter(xfmt)

        f.savefig(self._analyze_img_file, bbox_inches='tight', dpi=200)
        plt.close(f)

    def _plot_analyze_Pong(self):
        import IPython; IPython.embed()

        f, axes = plt.subplots(len(analyze_groups), 1, figsize=(20, 15), sharex=True, sharey=True)

        ### plot cum reward versus time step
        for ax, analyze_group in zip(axes.ravel(), analyze_groups):
            data_interp = DataAverageInterpolation()
            min_step = max_step = None
            for i, analyze in enumerate(analyze_group):
                rollouts = list(itertools.chain(*analyze.eval_rollouts_itrs))
                rollouts = sorted(rollouts, key=lambda r: r['steps'][0])
                steps, cum_rewards = [], []
                for r in rollouts:
                    steps.append(r['steps'][-1])
                    cum_rewards.append(np.sum(r['rewards']))

                def moving_avg_std(idxs, data, window):
                    avg_idxs, means, stds = [], [], []
                    for i in range(window, len(data)):
                        avg_idxs.append(np.mean(idxs[i - window:i]))
                        means.append(np.mean(data[i - window:i]))
                        stds.append(np.std(data[i - window:i]))
                    return avg_idxs, np.asarray(means), np.asarray(stds)

                steps, cum_rewards, _ = moving_avg_std(steps, cum_rewards, window=100)

                ax.plot(steps, cum_rewards, color=analyze.plot['color'],
                        alpha=np.linspace(1., 0.4, len(analyze_group))[i])

            #     data_interp.add_data(steps, cum_rewards)
            #     if min_step is None:
            #         min_step = steps[0]
            #     if max_step is None:
            #         max_step = steps[-1]
            #     min_step = max(min_step, steps[0])
            #     max_step = min(max_step, steps[-1])
            #
            # steps = np.r_[min_step:max_step:50.][1:-1]
            # cum_rewards_mean, cum_rewards_std = data_interp.eval(steps)
            #
            # ax.plot(steps, cum_rewards_mean, color=analyze.plot['color'], label=analyze.plot['label'])
            # ax.fill_between(steps, cum_rewards_mean - cum_rewards_std, cum_rewards_mean + cum_rewards_std,
            #                 color=analyze.plot['color'], alpha=0.4)

            ax.vlines(analyze.params['alg']['sample_after_n_steps'], -24, 24, color='k', linestyle='--')
            ax.set_ylim((-24, 24))
            ax.grid()
            # ax.set_title(analyze.name)
            ax.set_ylabel('Cumreward')
            ax.legend(loc='upper right')

        ### for all plots
        ax.set_xlabel('Steps')
        xfmt = ticker.ScalarFormatter()
        xfmt.set_powerlimits((0, 0))
        ax.xaxis.set_major_formatter(xfmt)

        f.savefig(self._analyze_img_file, bbox_inches='tight', dpi=200)
        plt.close(f)

    def _plot_analyze_InvertedPendulum(self):
        f, axes = create_best_fit_axes(len(self._analyze_groups), figsize=(15, 15))

        ### plot cum reward versus time step
        for ax, analyze_group in zip(axes.ravel(), self._analyze_groups):
            data_interp = DataAverageInterpolation()
            min_step = max_step = None
            for analyze in analyze_group:
                rollouts = list(itertools.chain(*analyze.train_rollouts_itrs))
                rollouts = sorted(rollouts, key=lambda r: r['steps'][-1])
                steps, cum_rewards = [], []
                for r in rollouts:
                    steps.append(r['steps'][-1])
                    cum_rewards.append(np.sum(r['rewards']))

                def moving_avg_std(idxs, data, window):
                    avg_idxs, means, stds = [], [], []
                    for i in range(window, len(data)):
                        avg_idxs.append(np.mean(idxs[i - window:i]))
                        means.append(np.mean(data[i - window:i]))
                        stds.append(np.std(data[i - window:i]))
                    return avg_idxs, np.asarray(means), np.asarray(stds)

                steps, cum_rewards, _ = moving_avg_std(steps, cum_rewards, window=50)

                data_interp.add_data(steps, cum_rewards)
                if min_step is None:
                    min_step = steps[0]
                if max_step is None:
                    max_step = steps[-1]
                min_step = max(min_step, steps[0])
                max_step = min(max_step, steps[-1])

            steps = np.r_[min_step:max_step:50.][1:-1]
            cum_rewards_mean, cum_rewards_std = data_interp.eval(steps)

            ax.plot(steps, cum_rewards_mean, color=analyze.plot['color'], label=analyze.plot['label'])
            ax.fill_between(steps, cum_rewards_mean - cum_rewards_std, cum_rewards_mean + cum_rewards_std,
                            color=analyze.plot['color'], alpha=0.4)

            max_path_length = analyze_group[0].params['alg']['max_path_length'] \
                if 'max_path_length' in analyze_group[0].params['alg'] \
                else analyze_group[0].env.horizon
            ax.set_ylim((-0.1*max_path_length, 1.1*max_path_length))
            ax.grid()
            ax.set_title(analyze_group[0].plot['label'], {'fontsize': 10})
            ax.set_ylabel('Cumulative reward')
            ax.set_xlabel('Steps')
            xfmt = ticker.ScalarFormatter()
            xfmt.set_powerlimits((0, 0))
            ax.xaxis.set_major_formatter(xfmt)

        plt.tight_layout()
        f.savefig(self._analyze_img_file, bbox_inches='tight', dpi=200)
        plt.close(f)

    def _plot_analyze_InvertedPendulum_offpolicy(self):
        f, axes = create_best_fit_axes(len(self._analyze_groups), figsize=(15, 15))

        ### plot cum reward versus time step
        for ax, analyze_group in zip(axes.ravel(), self._analyze_groups):
            data_interp = DataAverageInterpolation()
            data_interp_cummax = DataAverageInterpolation()
            min_step = max_step = None
            for analyze in analyze_group:
                assert('offpolicy' in analyze.params['alg'])
                steps = np.array(analyze.progress['Step'], dtype=np.float32)
                onpolicy_switch_idx = np.convolve(steps, [-1, 1], 'same').argmax()
                steps[onpolicy_switch_idx:] += analyze.params['alg']['offpolicy']['total_steps']
                
                cum_rewards = analyze.progress['CumRewardMean']
                cum_rewards_cummax = cum_rewards.cummax()
           
                data_interp.add_data(steps, cum_rewards)
                data_interp_cummax.add_data(steps, cum_rewards_cummax)
                if min_step is None:
                    min_step = steps[0]
                if max_step is None:
                    max_step = steps[-1]
                min_step = max(min_step, steps[0])
                max_step = min(max_step, steps[-1])

            steps = np.r_[min_step:max_step:50.][1:-1]
            cum_rewards_mean, cum_rewards_std = data_interp.eval(steps)
            cum_rewards_cummax_mean, cum_rewards_cummax_std = data_interp_cummax.eval(steps)

            ax.plot(steps, cum_rewards_mean, color='k', label=analyze.plot['label'])
            ax.fill_between(steps, cum_rewards_mean - cum_rewards_std, cum_rewards_mean + cum_rewards_std, color='k', alpha=0.4)
            
            ax.plot(steps, cum_rewards_cummax_mean, color='r', label=analyze.plot['label'])
            ax.fill_between(steps, cum_rewards_cummax_mean - cum_rewards_cummax_std, cum_rewards_cummax_mean + cum_rewards_cummax_std, color='r', alpha=0.4)

            max_path_length = analyze_group[0].params['alg']['max_path_length'] \
                if 'max_path_length' in analyze_group[0].params['alg'] \
                else analyze_group[0].env.horizon
            ax.vlines(analyze.params['alg']['offpolicy']['total_steps'], -0.1*max_path_length, 1.1*max_path_length, color='k', linestyle='--')
            ax.set_ylim((-0.1*max_path_length, 1.1*max_path_length))
            ax.grid()
            ax.set_title(analyze_group[0].plot['label'], {'fontsize': 10})
            ax.set_ylabel('Cumulative reward')
            ax.set_xlabel('Steps')
            xfmt = ticker.ScalarFormatter()
            xfmt.set_powerlimits((0, 0))
            ax.xaxis.set_major_formatter(xfmt)

        plt.tight_layout()
        f.savefig(self._analyze_img_file, bbox_inches='tight', dpi=200)
        plt.close(f)

    def _plot_analyze_Reacher(self):
        import IPython; IPython.embed()

        f, axes = create_best_fit_axes(len(self._analyze_groups), figsize=(15, 15))

        ### plot cum reward versus time step
        for ax, analyze_group in zip(axes.ravel(), self._analyze_groups):
            data_interp = DataAverageInterpolation()
            min_step = max_step = None
            for analyze in analyze_group:
                rollouts = list(itertools.chain(*analyze.eval_rollouts_itrs))
                rollouts = sorted(rollouts, key=lambda r: r['steps'][-1])
                steps, cum_rewards = [], []
                for r in rollouts:
                    steps.append(r['steps'][-1])
                    cum_rewards.append(np.sum(r['rewards']))

                def moving_avg_std(idxs, data, window):
                    avg_idxs, means, stds = [], [], []
                    for i in range(window, len(data)):
                        avg_idxs.append(np.mean(idxs[i - window:i]))
                        means.append(np.mean(data[i - window:i]))
                        stds.append(np.std(data[i - window:i]))
                    return avg_idxs, np.asarray(means), np.asarray(stds)

                steps, cum_rewards, _ = moving_avg_std(steps, cum_rewards, window=50)

                data_interp.add_data(steps, cum_rewards)
                if min_step is None:
                    min_step = steps[0]
                if max_step is None:
                    max_step = steps[-1]
                min_step = max(min_step, steps[0])
                max_step = min(max_step, steps[-1])

            steps = np.r_[min_step:max_step:50.][1:-1]
            cum_rewards_mean, cum_rewards_std = data_interp.eval(steps)

            ax.plot(steps, cum_rewards_mean, color=analyze.plot['color'], label=analyze.plot['label'])
            ax.fill_between(steps, cum_rewards_mean - cum_rewards_std, cum_rewards_mean + cum_rewards_std,
                            color=analyze.plot['color'], alpha=0.4)

            ax.grid()
            ax.set_title(analyze_group[0].plot['label'], {'fontsize': 10})
            ax.set_ylabel('Cumulative reward')
            ax.set_xlabel('Steps')
            xfmt = ticker.ScalarFormatter()
            xfmt.set_powerlimits((0, 0))
            ax.xaxis.set_major_formatter(xfmt)

        ymin, ymax = np.inf, -np.inf
        xmin, xmax = np.inf, -np.inf
        for ax in axes.ravel():
            ymin = min(ymin, ax.get_ylim()[0])
            ymax = max(ymax, ax.get_ylim()[1])
            xmin = min(xmin, ax.get_xlim()[0])
            xmax = max(xmax, ax.get_xlim()[1])

        for ax in axes.ravel():
            ax.set_ylim((ymin, ymax))
            ax.set_xlim((xmin, xmax))

        plt.tight_layout()
        f.savefig(self._analyze_img_file, bbox_inches='tight', dpi=200)
        plt.close(f)

    def _plot_analyze_HalfCheetah(self):
        import IPython; IPython.embed()

        f, axes = create_best_fit_axes(len(self._analyze_groups), figsize=(15, 15), sharex=True, sharey=True)

        ### plot cum reward versus time step
        for ax, analyze_group in zip(axes.ravel(), self._analyze_groups):
            data_interp = DataAverageInterpolation()
            min_step = max_step = None
            for analyze in analyze_group:
                rollouts = list(itertools.chain(*analyze.eval_rollouts_itrs))
                rollouts = sorted(rollouts, key=lambda r: r['steps'][-1])
                steps, cum_rewards = [], []
                for r in rollouts:
                    steps.append(r['steps'][-1])
                    cum_rewards.append(np.sum(r['rewards']))

                def moving_avg_std(idxs, data, window):
                    avg_idxs, means, stds = [], [], []
                    for i in range(window, len(data)):
                        avg_idxs.append(np.mean(idxs[i - window:i]))
                        means.append(np.mean(data[i - window:i]))
                        stds.append(np.std(data[i - window:i]))
                    return avg_idxs, np.asarray(means), np.asarray(stds)

                steps, cum_rewards, _ = moving_avg_std(steps, cum_rewards, window=50)

                data_interp.add_data(steps, cum_rewards)
                if min_step is None:
                    min_step = steps[0]
                if max_step is None:
                    max_step = steps[-1]
                min_step = max(min_step, steps[0])
                max_step = min(max_step, steps[-1])

            steps = np.r_[min_step:max_step:50.][1:-1]
            cum_rewards_mean, cum_rewards_std = data_interp.eval(steps)

            ax.plot(steps, cum_rewards_mean, color=analyze.plot['color'], label=analyze.plot['label'])
            ax.fill_between(steps, cum_rewards_mean - cum_rewards_std, cum_rewards_mean + cum_rewards_std,
                            color=analyze.plot['color'], alpha=0.4)

            ax.grid()
            ax.set_title(analyze_group[0].plot['label'], {'fontsize': 10})
            ax.set_ylabel('Cumulative reward')
            ax.set_xlabel('Steps')
            xfmt = ticker.ScalarFormatter()
            xfmt.set_powerlimits((0, 0))
            ax.xaxis.set_major_formatter(xfmt)

        plt.tight_layout()
        f.savefig(self._analyze_img_file, bbox_inches='tight', dpi=200)
        plt.close(f)

    def _plot_analyze_Pendulum(self):
        import IPython; IPython.embed()

        f, axes = create_best_fit_axes(len(self._analyze_groups), figsize=(15, 15), sharex=True, sharey=True)

        ### plot cum reward versus time step
        for ax, analyze_group in zip(axes.ravel(), self._analyze_groups):
            data_interp = DataAverageInterpolation()
            min_step = max_step = None
            for analyze in analyze_group:
                rollouts = list(itertools.chain(*analyze.eval_rollouts_itrs))
                rollouts = sorted(rollouts, key=lambda r: r['steps'][-1])
                steps, cum_rewards = [], []
                for r in rollouts:
                    steps.append(r['steps'][-1])
                    cum_rewards.append(np.sum(r['rewards']))

                def moving_avg_std(idxs, data, window):
                    avg_idxs, means, stds = [], [], []
                    for i in range(window, len(data)):
                        avg_idxs.append(np.mean(idxs[i - window:i]))
                        means.append(np.mean(data[i - window:i]))
                        stds.append(np.std(data[i - window:i]))
                    return avg_idxs, np.asarray(means), np.asarray(stds)

                steps, cum_rewards, _ = moving_avg_std(steps, cum_rewards, window=50)

                data_interp.add_data(steps, cum_rewards)
                if min_step is None:
                    min_step = steps[0]
                if max_step is None:
                    max_step = steps[-1]
                min_step = max(min_step, steps[0])
                max_step = min(max_step, steps[-1])

            steps = np.r_[min_step:max_step:50.][1:-1]
            cum_rewards_mean, cum_rewards_std = data_interp.eval(steps)

            ax.plot(steps, cum_rewards_mean, color=analyze.plot['color'], label=analyze.plot['label'])
            ax.fill_between(steps, cum_rewards_mean - cum_rewards_std, cum_rewards_mean + cum_rewards_std,
                            color=analyze.plot['color'], alpha=0.4)

            ax.grid()
            ax.set_title(analyze_group[0].plot['label'], {'fontsize': 7})
            ax.set_ylabel('Cumulative reward')
            ax.set_xlabel('Steps')
            xfmt = ticker.ScalarFormatter()
            xfmt.set_powerlimits((0, 0))
            ax.xaxis.set_major_formatter(xfmt)

        plt.tight_layout()
        f.savefig(self._analyze_img_file, bbox_inches='tight', dpi=200)
        plt.close(f)

    def _plot_analyze_CartPoleSwingup(self):
        import IPython; IPython.embed()

        f, axes = create_best_fit_axes(len(self._analyze_groups), figsize=(15, 15), sharex=True, sharey=True)

        ### plot cum reward versus time step
        for ax, analyze_group in zip(axes.ravel(), self._analyze_groups):
            data_interp = DataAverageInterpolation()
            min_step = max_step = None
            for analyze in analyze_group:
                rollouts = list(itertools.chain(*analyze.eval_rollouts_itrs))
                rollouts = sorted(rollouts, key=lambda r: r['steps'][-1])
                steps = [r['steps'][0] for r in rollouts]
                thetas = np.abs([np.arccos(r['observations'][-1, 2]) * 180. / np.pi for r in rollouts])

                def moving_avg_std(idxs, data, window):
                    avg_idxs, means, stds = [], [], []
                    for i in range(window, len(data)):
                        avg_idxs.append(np.mean(idxs[i - window:i]))
                        means.append(np.mean(data[i - window:i]))
                        stds.append(np.std(data[i - window:i]))
                    return avg_idxs, np.asarray(means), np.asarray(stds)

                steps, thetas, _ = moving_avg_std(steps, thetas, window=20)

                data_interp.add_data(steps, thetas)
                if min_step is None:
                    min_step = steps[0]
                if max_step is None:
                    max_step = steps[-1]
                min_step = max(min_step, steps[0])
                max_step = min(max_step, steps[-1])

            steps = np.r_[min_step:max_step:50.][1:-1]
            thetas_mean, thetas_std = data_interp.eval(steps)

            ax.plot(steps, thetas_mean, color=analyze.plot['color'], label=analyze.plot['label'])
            ax.fill_between(steps, thetas_mean - thetas_std, thetas_mean + thetas_std,
                            color=analyze.plot['color'], alpha=0.4)

            ax.grid()
            ax.set_title(analyze_group[0].plot['label'], {'fontsize': 7})
            ax.set_ylabel('theta')
            ax.set_xlabel('Steps')
            xfmt = ticker.ScalarFormatter()
            xfmt.set_powerlimits((0, 0))
            ax.xaxis.set_major_formatter(xfmt)

        plt.tight_layout()
        f.savefig(self._analyze_img_file, bbox_inches='tight', dpi=200)
        plt.close(f)

    def _plot_analyze_Hopper(self):
        import IPython; IPython.embed()

        f, axes = plt.subplots(len(self._analyze_groups), 1, figsize=(15, 15), sharex=True, sharey=True)

        ### plot cum reward versus time step
        for ax, analyze_group in zip(axes.ravel(), self._analyze_groups):
            for analyze in analyze_group:
                rollouts = list(itertools.chain(*analyze.eval_rollouts_itrs))
                rollouts = sorted(rollouts, key=lambda r: r['steps'][0])
                steps, cum_rewards = [], []
                for r in rollouts:
                    steps.append(r['steps'][-1])
                    cum_rewards.append(np.sum(r['rewards']))

                def moving_avg_std(idxs, data, window):
                    avg_idxs, means, stds = [], [], []
                    for i in range(window, len(data)):
                        avg_idxs.append(np.mean(idxs[i - window:i]))
                        means.append(np.mean(data[i - window:i]))
                        stds.append(np.std(data[i - window:i]))
                    return avg_idxs, np.asarray(means), np.asarray(stds)

                steps, cum_rewards_mean, cum_rewards_std = moving_avg_std(steps, cum_rewards, window=500)

                ax.plot(steps, cum_rewards_mean, color=analyze.plot['color'], label=analyze.plot['label'])
                ax.fill_between(steps, cum_rewards_mean - cum_rewards_std, cum_rewards_mean + cum_rewards_std,
                                color=analyze.plot['color'], alpha=0.4)

            ax.grid()
            ax.set_title(analyze_group[0].plot['label'], {'fontsize': 10})
            ax.set_ylabel('Cumulative reward')
            ax.set_xlabel('Steps')
            xfmt = ticker.ScalarFormatter()
            xfmt.set_powerlimits((0, 0))
            ax.xaxis.set_major_formatter(xfmt)

        plt.tight_layout()
        f.savefig(self._analyze_img_file, bbox_inches='tight', dpi=200)
        plt.close(f)

    def _plot_analyze_Tennis(self):
        f, axes = plt.subplots(1 + len(self._analyze_groups), 1, figsize=(15, 5 * len(self._analyze_groups)),
                               sharex=True)

        ### plot training cost
        ax = axes.ravel()[0]
        for analyze_group in self._analyze_groups:
            data_interp = DataAverageInterpolation()
            min_step = max_step = None
            for analyze in analyze_group:
                steps = np.array(analyze.progress['Step'])
                costs = np.array(analyze.progress['Cost'])
                data_interp.add_data(steps, costs)

                if min_step is None:
                    min_step = steps[0]
                if max_step is None:
                    max_step = steps[-1]
                min_step = max(min_step, steps[0])
                max_step = min(max_step, steps[-1])

            steps = np.r_[min_step:max_step:1.]
            costs_mean, costs_std = data_interp.eval(steps)

            ax.plot(steps, costs_mean, color=analyze.plot['color'], label=analyze.plot['label'])
            ax.fill_between(steps, costs_mean - costs_std, costs_mean + costs_std,
                            color=analyze.plot['color'], alpha=0.4)
        ax.set_ylabel('Training cost')
        ax.legend(loc='upper right')

        ### plot cum reward versus time step
        for ax, analyze_group in zip(axes.ravel()[1:], self._analyze_groups):
            data_interp = DataAverageInterpolation()
            min_step = max_step = None
            for analyze in analyze_group:
                rollouts = list(itertools.chain(*analyze.train_rollouts_itrs))
                rollouts = sorted(rollouts, key=lambda r: r['steps'][-1])
                steps, cum_rewards = [], []
                for r in rollouts:
                    steps.append(r['steps'][-1])
                    cum_rewards.append(np.sum(r['rewards']))

                def moving_avg_std(idxs, data, window):
                    avg_idxs, means, stds = [], [], []
                    for i in range(window, len(data)):
                        avg_idxs.append(np.mean(idxs[i - window:i]))
                        means.append(np.mean(data[i - window:i]))
                        stds.append(np.std(data[i - window:i]))
                    return avg_idxs, np.asarray(means), np.asarray(stds)

                steps, cum_rewards, _ = moving_avg_std(steps, cum_rewards, window=50)

                data_interp.add_data(steps, cum_rewards)
                if min_step is None:
                    min_step = steps[0]
                if max_step is None:
                    max_step = steps[-1]
                min_step = max(min_step, steps[0])
                max_step = min(max_step, steps[-1])

            steps = np.r_[min_step:max_step:50.][1:-1]
            cum_rewards_mean, cum_rewards_std = data_interp.eval(steps)

            ax.plot(steps, cum_rewards_mean, color=analyze.plot['color'], label=analyze.plot['label'])
            ax.fill_between(steps, cum_rewards_mean - cum_rewards_std, cum_rewards_mean + cum_rewards_std,
                            color=analyze.plot['color'], alpha=0.4)
            ax.set_ylim((-22, 22))
            ax.grid()
            ax.set_title(' '.join([analyze.name for analyze in analyze_group]))
        ax.set_ylabel('Cumulative reward')

        ### for all plots
        ax.set_xlabel('Steps')
        xfmt = ticker.ScalarFormatter()
        xfmt.set_powerlimits((0, 0))
        ax.xaxis.set_major_formatter(xfmt)

        f.savefig(self._analyze_img_file, bbox_inches='tight', dpi=200)
        plt.close(f)



    ###########
    ### Run ###
    ###########

    def run(self):
        self._plot_analyze()

if __name__ == '__main__':
    SAVE_FOLDER = '/media/gkahn/ExtraDrive1/rllab/rnn_critic/'

    analyze_groups = []
    for start in range(228, 245, 3):
        analyze_group = []
        for i in range(start, start + 3):
            print('\nphd{0:03d}\n'.format(i))
            # try:
            analyze = AnalyzeRNNCritic(os.path.join(SAVE_FOLDER, 'phd{0:03d}'.format(i)),
                                       plot={
                                           'label': '',
                                           'color': 'k',
                                       },
                                       clear_obs=True,
                                       create_new_envs=False)
            analyze.plot['label'] = 'N: {0}, H: {1}, r_adviser: {2}'.format(
                analyze.params['policy']['N'],
                analyze.params['policy']['H'],
                float(analyze.params['alg']['env'].split('r_adviser=')[-1].split(',')[0])
            )
            analyze_group.append(analyze)
            # except:
            #     print('failed')

        analyze_groups.append(analyze_group)

    plotter = PlotAnalyzeRNNCritic(os.path.join(SAVE_FOLDER, 'analyze'), 'phd_228_245', analyze_groups)
    plotter.run()

