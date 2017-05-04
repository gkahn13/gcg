import argparse, os, sys
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

from rllab.misc.ext import set_seed
import rllab.misc.logger as logger
# from rllab.misc import tensor_utils
from sandbox.rocky.tf.misc import tensor_utils

from sandbox.gkahn.rnn_critic.envs.point_env import PointEnv
from sandbox.gkahn.rnn_critic.envs.sparse_point_env import SparsePointEnv
from sandbox.gkahn.rnn_critic.envs.chain_env import ChainEnv
from sandbox.gkahn.rnn_critic.policies.mac_policy import MACPolicy
from sandbox.gkahn.rnn_critic.sampler.vectorized_rollout_sampler import RNNCriticVectorizedRolloutSampler
from sandbox.gkahn.rnn_critic.sampler.sampler import RNNCriticSampler

### environments
from sandbox.rocky.tf.envs.base import TfEnv
from rllab.envs.normalized_env import normalize
from rllab.envs.gym_env import GymEnv
import gym
from sandbox.gkahn.rnn_critic.envs.atari_wrappers import wrap_deepmind
from sandbox.gkahn.rnn_critic.envs.pygame_wrappers import wrap_pygame
from rllab.envs.gym_env import GymEnv
from sandbox.gkahn.rnn_critic.envs.premade_gym_env import PremadeGymEnv
try:
    import gym_ple
except:
    pass
from sandbox.gkahn.rnn_critic.envs.point_env import PointEnv
from sandbox.gkahn.rnn_critic.envs.sparse_point_env import SparsePointEnv
from sandbox.gkahn.rnn_critic.envs.chain_env import ChainEnv

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
        a, agent_info = agent.get_action(o)
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
    def __init__(self, folder, skip_itr=1, max_itr=sys.maxsize, plot=dict(), create_new_envs=True, clear_obs=False):
        """
        :param kwargs: holds random extra properties
        """
        self._folder = folder
        self._skip_itr = skip_itr
        self._max_itr = max_itr
        self._create_new_envs = create_new_envs
        self._clear_obs = clear_obs

        ### load data
        logger.log('AnalyzeRNNCritic: Loading data')
        self.name = os.path.basename(self._folder)
        self.plot = plot
        logger.log('AnalyzeRNNCritic: params_file: {0}'.format(self._params_file))
        with open(self._params_file, 'r') as f:
            self.params = yaml.load(f)
        logger.log('AnalyzeRNNCritic: Loading csv')
        try:
            self.progress = pandas.read_csv(self._progress_file)
        except Exception as e:
            logger.log('Could not open csv: {0}'.format(str(e)))
            self.progress = None
        logger.log('AnalyzeRNNCritic: Loaded csv')

        self.train_rollouts_itrs = self._load_rollouts_itrs()
        self.eval_rollouts_itrs = self._load_rollouts_itrs(eval=True)
        logger.log('AnalyzeRNNCritic: Loaded all itrs')
        self.env = TfEnv(normalize(eval(self.params['alg']['env'])))
        logger.log('AnalyzeRNNCritic: Created env')
        # self.eval_rollouts_itrs = []
        # if not os.path.exists(self._eval_rollouts_itrs_file):
        #     eval_rollouts_itrs = self._eval_all_policies()
        #     logger.log('AnalyzeRNNCritic: Eval all policies')
        #     with open(self._eval_rollouts_itrs_file, 'wb') as f:
        #         pickle.dump(eval_rollouts_itrs, f)
        #     logger.log('AnalyzeRNNCritic: Loaded eval rollouts')
        # with open(self._eval_rollouts_itrs_file, 'rb') as f:
        #     self.eval_rollouts_itrs = pickle.load(f)
        logger.log('AnalyzeRNNCritic: Finished loading data')

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
        yamls = [fname for fname in os.listdir(self._folder) if os.path.splitext(fname)[-1] == '.yaml']
        assert(len(yamls) == 1)
        return os.path.join(self._folder, yamls[0])

    @property
    def _analyze_img_file(self):
        return os.path.join(self._folder, 'analyze.png')

    def _analyze_rollout_img_file(self, itr, is_train):
        return os.path.join(self._folder, 'analyze_{0}_rollout_itr_{1:d}.png'.format('train' if is_train else 'eval', itr))

    def _analyze_policy_img_file(self, itr):
        if itr is not None:
            return os.path.join(self._folder, 'analyze_policy_itr_{0:d}.png'.format(itr))
        else:
            return os.path.join(self._folder, 'analyze_policy.png')

    @property
    def _analyze_value_function_img_file(self):
        return os.path.join(self._folder, 'analyze_value_function.png')

    def _analyze_Q_function_img_file(self, itr):
        return os.path.join(self._folder, 'analyze_Q_function_itr{0:d}.png'.format(itr))

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

    def _eval_all_policies(self):
        rollouts_itrs = []
        itr = 0
        while os.path.exists(self._itr_file(itr)) and itr < self._max_itr:
            # set seed
            if self.params['seed'] is not None:
                set_seed(self.params['seed'])
                inner_env(self.env).env.seed(self.params['seed'])

            sess, graph = MACPolicy.create_session_and_graph(gpu_device=self.params['policy']['gpu_device'],
                                                             gpu_frac=self.params['policy']['gpu_frac'])
            with graph.as_default(), sess.as_default():
                policy = self._load_itr_policy(itr)

                logger.log('Evaluating policy for itr {0}'.format(itr))
                n_envs = 1
                if 'max_path_length' in self.params['alg']:
                    max_path_length = self.params['alg']['max_path_length']
                else:
                    max_path_length = self.env.horizon
                sampler = RNNCriticSampler(
                    policy=policy,
                    env=self.env,
                    n_envs=n_envs,
                    replay_pool_size=int(1e4),
                    max_path_length=max_path_length,
                    save_rollouts=True
                )
                rollouts = []
                step = 0
                while len(rollouts) < 25:
                    sampler.step(step)
                    step += n_envs
                    rollouts += sampler.get_recent_paths()

            rollouts_itrs.append(rollouts)
            itr += self._skip_itr

        return rollouts_itrs

    ################
    ### Plotting ###
    ################

    def _plot_analyze(self, train_rollouts_itrs, eval_rollouts_itrs):
        env = inner_env(self.env)
        if isinstance(env, ChainEnv):
            self._plot_analyze_ChainEnv(train_rollouts_itrs, eval_rollouts_itrs)
        elif isinstance(env, SparsePointEnv):
            self._plot_analyze_PointEnv(train_rollouts_itrs, eval_rollouts_itrs)
        elif isinstance(env, GymEnv):
            self._plot_analyze_Gym(train_rollouts_itrs, eval_rollouts_itrs)
        else:
            self._plot_analyze_general(train_rollouts_itrs, eval_rollouts_itrs)

    def _plot_analyze_general(self, train_rollouts_itrs, eval_rollouts_itrs):
        f, axes = plt.subplots(5, 1, figsize=(2 * len(train_rollouts_itrs), 7.5), sharex=True)
        f.tight_layout()

        ### plot training cost
        ax = axes[0]
        costs = self.progress['Cost'][1:]
        steps = self.progress['Step'][1:]
        ax.plot(steps, costs, 'k-')
        ax.set_ylabel('Cost')

        ### plot avg reward
        ax = axes[1]
        avg_reward_means = self.progress['AvgRewardMean']
        avg_reward_stds = self.progress['AvgRewardStd']
        steps = self.progress['Step']
        ax.plot(steps, avg_reward_means, 'k-')
        ax.fill_between(steps, avg_reward_means - avg_reward_stds, avg_reward_means + avg_reward_stds,
                        color='k', alpha=0.4)
        ax.set_ylabel('Average reward')

        ### plot final reward
        ax = axes[2]
        final_reward_means = self.progress['FinalRewardMean']
        final_reward_stds = self.progress['FinalRewardStd']
        steps = self.progress['Step']
        ax.plot(steps, final_reward_means, 'k-')
        ax.fill_between(steps, final_reward_means - final_reward_stds, final_reward_means + final_reward_stds,
                        color='k', alpha=0.4)
        ax.set_ylabel('Final reward')

        start_step = self.params['alg']['learn_after_n_steps']
        end_step = self.params['alg']['total_steps']
        save_step = self.params['alg']['save_every_n_steps']
        first_save_step = save_step * np.floor(start_step / float(save_step))
        itr_steps = np.r_[first_save_step:end_step:save_step]

        def plot_reward(ax, rewards):
            color = 'k'
            bp = ax.boxplot(rewards,
                            positions=itr_steps,
                            widths=0.4 * self._skip_itr * save_step)
            for key in ('boxes', 'medians', 'whiskers', 'fliers', 'caps'):
                plt.setp(bp[key], color=color)
            for cap_line, median_line in zip(bp['caps'][1::2], bp['medians']):
                cx, cy = cap_line.get_xydata()[1]  # top of median line
                mx, my = median_line.get_xydata()[1]
                ax.text(cx, cy, '%.2f' % my,
                        horizontalalignment='right',
                        verticalalignment='bottom',
                        color='r')  # draw above, centered
            # for line in bp['medians']:
            #     # get position data for median line
            #     x, y = line.get_xydata()[1]  # top of median line
            #     # overlay median value
            #     ax.text(x, y, '%.2f' % y,
            #             horizontalalignment='left',
            #             verticalalignment='center',
            #             color='r')  # draw above, centered
            # for line in bp['boxes']:
            #     x, y = line.get_xydata()[0]  # bottom of left line
            #     ax.text(x, y, '%.2f' % y,
            #             horizontalalignment='right',  # centered
            #             verticalalignment='center',
            #             color='k', alpha=0.5)  # below
            #     x, y = line.get_xydata()[3]  # bottom of right line
            #     ax.text(x, y, '%.2f' % y,
            #             horizontalalignment='right',  # centered
            #             verticalalignment='center',
            #             color='k', alpha=0.5)  # below
            plt.setp(bp['fliers'], marker='_')
            plt.setp(bp['fliers'], markeredgecolor=color)

        ### plot train final reward
        ax = axes[3]
        rewards = [[rollout['rewards'][-1] for rollout in rollouts] for rollouts in train_rollouts_itrs]
        plot_reward(ax, rewards)
        ax.set_ylabel('Train final reward')

        ### plot eval final reward
        ax = axes[4]
        rewards = [[rollout['rewards'][-1] for rollout in rollouts] for rollouts in eval_rollouts_itrs]
        plot_reward(ax, rewards)
        ax.set_ylabel('Eval final reward')
        ax.set_xlabel('Steps')
        xfmt = ticker.ScalarFormatter()
        xfmt.set_powerlimits((0, 0))
        ax.xaxis.set_major_formatter(xfmt)

        ### for all
        for ax in axes:
            ax.set_xlim((-save_step/2., end_step))
            ax.set_xticks(itr_steps)

        f.savefig(self._analyze_img_file, bbox_inches='tight')
        plt.close(f)

    def _plot_analyze_ChainEnv(self, train_rollouts_itrs, eval_rollouts_itrs):
        num_steps = sum([len(r['observations']) for r in itertools.chain(*train_rollouts_itrs)])
        f, axes = plt.subplots(3, 1, figsize=(2. * num_steps / 500., 7.5), sharex=True)
        f.tight_layout()

        ### plot training cost
        ax = axes[0]
        costs = self.progress['Cost'][1:]
        steps = self.progress['Step'][1:]
        ax.plot(steps, costs, 'k-')
        ax.set_ylabel('Cost')

        ### plot training rollout length vs step
        ax = axes[1]
        rollouts = list(itertools.chain(*train_rollouts_itrs))
        rollout_lens = [len(r['observations']) for r in rollouts]
        steps = [r['steps'][-1] for r in rollouts]
        ax.plot(steps, rollout_lens, color='k', marker='|', linestyle='', markersize=10.)
        ax.vlines(self.params['alg']['learn_after_n_steps'], 0, ax.get_ylim()[1], colors='g', linestyles='dashed')
        ax.hlines(self.env.spec.observation_space.n, steps[0], steps[-1], colors='r', linestyles='dashed')
        ax.set_ylabel('Rollout length')

        ### plot training rollout length vs step smoothed
        ax = axes[2]
        def moving_avg_std(idxs, data, window):
            means, stds = [], []
            for i in range(window, len(data)):
                means.append(np.mean(data[i-window:i]))
                stds.append(np.std(data[i - window:i]))
            return idxs[:-window], np.asarray(means), np.asarray(stds)
        moving_steps, rollout_lens_mean, rollout_lens_std = moving_avg_std(steps, rollout_lens, 5)
        ax.plot(moving_steps, rollout_lens_mean, 'k-')
        ax.fill_between(moving_steps, rollout_lens_mean - rollout_lens_std, rollout_lens_mean + rollout_lens_std,
                        color='k', alpha=0.4)
        ax.vlines(self.params['alg']['learn_after_n_steps'], 0, ax.get_ylim()[1], colors='g', linestyles='dashed')
        ax.hlines(self.env.spec.observation_space.n, steps[0], steps[-1], colors='r', linestyles='dashed')
        ax.set_ylabel('Rollout length')

        ### for all plots
        ax.set_xlabel('Steps')
        xfmt = ticker.ScalarFormatter()
        xfmt.set_powerlimits((0, 0))
        ax.xaxis.set_major_formatter(xfmt)

        f.savefig(self._analyze_img_file, bbox_inches='tight')
        plt.close(f)

    def _plot_analyze_PointEnv(self, train_rollouts_itrs, eval_rollouts_itrs):
        f, axes = plt.subplots(5, 1, figsize=(2 * len(train_rollouts_itrs), 7.5), sharex=True)
        f.tight_layout()

        ### plot training cost
        ax = axes[0]
        costs = self.progress['Cost'][1:]
        steps = self.progress['Step'][1:]
        ax.plot(steps, costs, 'k-')
        ax.set_ylabel('Cost')

        ### plot avg reward
        ax = axes[1]
        rollouts = list(itertools.chain(*train_rollouts_itrs))
        avg_rewards = [np.mean(r['rewards']) for r in rollouts]
        steps = [r['steps'][-1] for r in rollouts]
        ax.plot(steps, avg_rewards, color='k', linestyle='', marker='|', markersize=5.)
        ax.hlines(0, steps[0], steps[-1], colors='r', linestyles='dashed')
        ax.set_ylabel('Average reward')

        ### plot final reward
        ax = axes[2]
        final_rewards = [r['rewards'][-1] for r in rollouts]
        ax.plot(steps, final_rewards, color='k', linestyle='', marker='|', markersize=5.)
        ax.hlines(0, steps[0], steps[-1], colors='r', linestyles='dashed')
        ax.set_ylabel('Final reward')

        start_step = self.params['alg']['learn_after_n_steps']
        end_step = self.params['alg']['total_steps']
        save_step = self.params['alg']['save_every_n_steps']
        first_save_step = save_step * np.floor(start_step / float(save_step))
        itr_steps = np.r_[first_save_step:end_step:save_step]

        def plot_reward(ax, rewards):
            color = 'k'
            bp = ax.boxplot(rewards,
                            positions=itr_steps,
                            widths=0.4 * self._skip_itr * save_step)
            for key in ('boxes', 'medians', 'whiskers', 'fliers', 'caps'):
                plt.setp(bp[key], color=color)
            for cap_line, median_line in zip(bp['caps'][1::2], bp['medians']):
                cx, cy = cap_line.get_xydata()[1]  # top of median line
                mx, my = median_line.get_xydata()[1]
                ax.text(cx, cy, '%.2f' % my,
                        horizontalalignment='right',
                        verticalalignment='bottom',
                        color='r')  # draw above, centered
            # for line in bp['medians']:
            #     # get position data for median line
            #     x, y = line.get_xydata()[1]  # top of median line
            #     # overlay median value
            #     ax.text(x, y, '%.2f' % y,
            #             horizontalalignment='left',
            #             verticalalignment='center',
            #             color='r')  # draw above, centered
            # for line in bp['boxes']:
            #     x, y = line.get_xydata()[0]  # bottom of left line
            #     ax.text(x, y, '%.2f' % y,
            #             horizontalalignment='right',  # centered
            #             verticalalignment='center',
            #             color='k', alpha=0.5)  # below
            #     x, y = line.get_xydata()[3]  # bottom of right line
            #     ax.text(x, y, '%.2f' % y,
            #             horizontalalignment='right',  # centered
            #             verticalalignment='center',
            #             color='k', alpha=0.5)  # below
            plt.setp(bp['fliers'], marker='_')
            plt.setp(bp['fliers'], markeredgecolor=color)

        ### plot train final reward
        ax = axes[3]
        rewards = [[rollout['rewards'][-1] for rollout in rollouts] for rollouts in train_rollouts_itrs]
        plot_reward(ax, rewards)
        ax.set_ylabel('Train final reward')

        ### plot eval final reward
        ax = axes[4]
        rewards = [[rollout['rewards'][-1] for rollout in rollouts] for rollouts in eval_rollouts_itrs]
        plot_reward(ax, rewards)
        ax.set_ylabel('Eval final reward')
        ax.set_xlabel('Steps')
        xfmt = ticker.ScalarFormatter()
        xfmt.set_powerlimits((0, 0))
        ax.xaxis.set_major_formatter(xfmt)

        ### for all
        for ax in axes:
            ax.set_xlim((-save_step/2., end_step))
            ax.set_xticks(itr_steps)

        f.savefig(self._analyze_img_file, bbox_inches='tight')
        plt.close(f)

    def _plot_analyze_Gym(self, train_rollouts_itrs, eval_rollouts_itrs):
        f, axes = plt.subplots(4, 1, figsize=(2 * len(train_rollouts_itrs), 10), sharex=True)
        f.tight_layout()

        train_rollouts = sorted(list(itertools.chain(*train_rollouts_itrs)), key=lambda r: r['steps'][0])
        eval_rollouts = sorted(list(itertools.chain(*eval_rollouts_itrs)), key=lambda r: r['steps'][0])

        ### plot train cumreward
        ax = axes[0]
        steps = [r['steps'][0] for r in train_rollouts]
        cumrewards = [np.sum(r['rewards']) for r in train_rollouts]
        steps, cumrewards_mean, cumrewards_std  = moving_avg_std(steps, cumrewards, window=100)
        ax.plot(steps, cumrewards_mean, 'k-')
        ax.fill_between(steps, cumrewards_mean - cumrewards_std, cumrewards_mean + cumrewards_std,
                        color='k', alpha=0.4)
        ax.set_ylabel('Train cumreward')
        ax.grid()

        ### plot eval cumreward
        ax = axes[1]
        steps = [r['steps'][0] for r in eval_rollouts]
        cumrewards = [np.sum(r['rewards']) for r in eval_rollouts]
        ax.plot(steps, cumrewards, 'r|', markersize=10.)
        steps, cumrewards_mean, cumrewards_std = moving_avg_std(steps, cumrewards, window=20)
        ax.plot(steps, cumrewards_mean, 'k-')
        ax.fill_between(steps, cumrewards_mean - cumrewards_std, cumrewards_mean + cumrewards_std,
                        color='k', alpha=0.4)
        ax.set_ylabel('Eval cumreward')
        ax.grid()

        ### plot training cost
        ax = axes[2]
        costs = self.progress['Cost'][1:]
        steps = self.progress['Step'][1:]
        ax.plot(steps, costs, 'k-')
        ax.set_ylabel('Cost')

        ### plot value function difference
        ax = axes[3]
        est_values_avg_diff_mean = self.progress['EvalEstValuesAvgDiffMean']
        est_values_avg_diff_std = self.progress['EvalEstValuesAvgDiffStd']
        est_values_max_diff_mean = self.progress['EvalEstValuesMaxDiffMean']
        est_values_max_diff_std = self.progress['EvalEstValuesMaxDiffStd']
        steps = self.progress['Step']
        ax.plot(steps, est_values_avg_diff_mean, 'k-', label='Avg')
        ax.fill_between(steps,
                        est_values_avg_diff_mean - est_values_avg_diff_std,
                        est_values_avg_diff_mean + est_values_avg_diff_std,
                        color='k', alpha=0.4)
        ax.plot(steps, est_values_max_diff_mean, 'r-', label='Max')
        ax.fill_between(steps,
                        est_values_max_diff_mean - est_values_max_diff_std,
                        est_values_max_diff_mean + est_values_max_diff_std,
                        color='r', alpha=0.4)
        ax.set_ylabel('EstValueDiff')
        ax.legend(loc='upper left', ncol=2)

        ax.set_xlabel('Steps')
        xfmt = ticker.ScalarFormatter()
        xfmt.set_powerlimits((0, 0))
        ax.xaxis.set_major_formatter(xfmt)

        f.savefig(self._analyze_img_file, bbox_inches='tight')
        plt.close(f)

    def _plot_rollouts(self, train_rollouts_itrs, eval_rollouts_itrs, env_itrs, is_train, plot_prior):
        env = inner_env(env_itrs[0])
        if isinstance(env, SparsePointEnv):
            self._plot_rollouts_PointEnv(train_rollouts_itrs, eval_rollouts_itrs, env_itrs, is_train, plot_prior)
        elif isinstance(env, GymEnv):
            if 'Reacher' in env.env_id:
                self._plot_rollouts_Reacher(train_rollouts_itrs, eval_rollouts_itrs, env_itrs, is_train, plot_prior)
            elif 'Swimmer' in env.env_id:
                self._plot_rollouts_Swimmer(train_rollouts_itrs, eval_rollouts_itrs, env_itrs, is_train, plot_prior)
        else:
            pass

    def _plot_rollouts_PointEnv(self, train_rollouts_itrs, eval_rollouts_itrs, env_itrs, is_train, plot_prior):
        rollouts_itrs = train_rollouts_itrs if is_train else eval_rollouts_itrs

        max_itr = len(rollouts_itrs) * self._skip_itr
        itrs = np.r_[0:max_itr:self._skip_itr]

        start_step = self.params['alg']['learn_after_n_steps']
        end_step = self.params['alg']['total_steps']
        save_step = self.params['alg']['save_every_n_steps']
        first_save_step = save_step * np.floor(start_step / float(save_step))
        itr_steps = np.r_[first_save_step:end_step:save_step]

        for itr, rollouts in zip(itrs, rollouts_itrs):

            N_rollouts = 25
            rollouts = sorted(rollouts, key=lambda r: r['rewards'][-1], reverse=True)
            if len(rollouts) > N_rollouts:
                rollouts = rollouts[::int(np.ceil(len(rollouts)) / float(N_rollouts))]

            nrows = ncols = int(np.ceil(np.sqrt(len(rollouts))))
            f, axes = plt.subplots(nrows, ncols, figsize=(10, 10))

            all_positions = np.vstack([np.array(rollout['observations']) for rollout in rollouts])
            xlim = ylim = (all_positions.min(), all_positions.max())

            for ax, rollout in zip(axes.ravel(), sorted(rollouts, key=lambda r: r['rewards'][-1], reverse=True)):
                # plot all prior rollouts
                if plot_prior:
                    for train_rollout in itertools.chain(*train_rollouts_itrs[:itr + 1]):
                        train_positions = np.array(train_rollout['observations'])
                        ax.plot(train_positions[:, 0], train_positions[:, 1], color='b', marker='', linestyle='-',
                                alpha=0.2)

                # plot this rollout
                positions = np.array(rollout['observations'])
                ax.plot(positions[:, 0], positions[:, 1], color='k', marker='o', linestyle='-', markersize=0.2)
                ax.plot([0], [0], color='r', marker='x', markersize=5.)
                ax.plot([positions[0, 0]], [positions[0, 1]], color='g', marker='o', markersize=5.)
                ax.plot([positions[-1, 0]], [positions[-1, 1]], color='y', marker='o', markersize=5.)

                ax.set_xlim(xlim)
                ax.set_ylim(ylim)
                ax.set_title('{0:.2f}'.format(rollout['rewards'][-1]))

            suptitle = f.suptitle('Step %.2e' % itr_steps[itr], y=1.05)
            f.tight_layout()

            f.savefig(self._analyze_rollout_img_file(itr, is_train), bbox_inches='tight', dpi=200.,
                      bbox_extra_artsist=(suptitle,))
            plt.close(f)

    def _plot_rollouts_Reacher(self, train_rollouts_itrs, eval_rollouts_itrs, env_itrs, is_train, plot_prior):
        def get_rollout_positions(rollout):
            observations = np.array(rollout['observations'])
            goal_pos = observations[0, 4:6]
            positions = observations[:, -3:-1] + goal_pos
            return positions, goal_pos

        rollouts_itrs = train_rollouts_itrs if is_train else eval_rollouts_itrs

        max_itr = len(rollouts_itrs) * self._skip_itr
        itrs = np.r_[0:max_itr:self._skip_itr]

        for itr, rollouts in zip(itrs, rollouts_itrs):

            N_rollouts = 25
            rollouts = sorted(rollouts, key=lambda r: r['rewards'][-1], reverse=True)
            if len(rollouts) > N_rollouts:
                rollouts = rollouts[::int(np.ceil(len(rollouts)) / float(N_rollouts))]

            nrows = ncols = int(np.ceil(np.sqrt(len(rollouts))))
            f, axes = plt.subplots(nrows, ncols, figsize=(10, 10))
            xlim = ylim = (-0.25, 0.25)

            for ax, rollout in zip(axes.ravel(), sorted(rollouts, key=lambda r: r['rewards'][-1], reverse=True)):
                # plot all prior rollouts
                if plot_prior:
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
                ax.set_title('{0:.2f}'.format(rollout['rewards'][-1]))

            f.tight_layout()

            f.savefig(self._analyze_rollout_img_file(itr, is_train), bbox_inches='tight', dpi=200.)
            plt.close(f)

    def _plot_rollouts_Swimmer(self, train_rollouts_itrs, eval_rollouts_itrs, env_itrs, is_train, plot_prior):
        if is_train:
            return # can't plot train b/c rollouts dont contain states

        def get_rollout_positions(rollout):
            qpos = rollout['env_infos']['qpos']
            positions = qpos[:,:2,:].reshape((len(qpos), -1))
            return positions

        rollouts_itrs = train_rollouts_itrs if is_train else eval_rollouts_itrs

        max_itr = len(rollouts_itrs) * self._skip_itr
        itrs = np.r_[0:max_itr:self._skip_itr]

        for itr, rollouts in zip(itrs, rollouts_itrs):

            N_rollouts = 25
            rollouts = sorted(rollouts, key=lambda r: np.sum(r['rewards']), reverse=True)
            if len(rollouts) > N_rollouts:
                rollouts = rollouts[::int(np.ceil(len(rollouts)) / float(N_rollouts))]

            nrows = ncols = int(np.ceil(np.sqrt(len(rollouts))))
            f, axes = plt.subplots(nrows, ncols, figsize=(10, 10))

            for ax, rollout in zip(axes.ravel(), sorted(rollouts, key=lambda r: np.sum(r['rewards']), reverse=True)):
                # plot this rollout
                positions = get_rollout_positions(rollout)
                ax.plot(positions[:, 0], positions[:, 1], color='k', marker='o', linestyle='-', markersize=0.2)
                ax.plot([positions[0, 0]], [positions[0, 1]], color='g', marker='o', markersize=5.)
                ax.plot([positions[-1, 0]], [positions[-1, 1]], color='y', marker='o', markersize=5.)

                ax.set_title('{0:.2f}'.format(np.sum(rollout['rewards'])))

            xlim = [np.inf, -np.inf]
            ylim = [np.inf, -np.inf]
            for ax in axes.ravel():
                xlim[0] = min(xlim[0], ax.get_xlim()[0])
                xlim[1] = max(xlim[1], ax.get_xlim()[1])
                ylim[0] = min(ylim[0], ax.get_ylim()[0])
                ylim[1] = max(ylim[1], ax.get_ylim()[1])
            for ax in axes.ravel():
                ax.set_xlim(xlim)
                ax.set_ylim(ylim)

            f.tight_layout()

            f.savefig(self._analyze_rollout_img_file(itr, is_train), bbox_inches='tight', dpi=200.)
            plt.close(f)

    def _plot_policies(self, rollouts_itrs, env_itrs):
        env = env_itrs[0]
        while hasattr(env, 'wrapped_env'):
            env = env.wrapped_env
        if isinstance(env, PointEnv):
            self._plot_policies_PointEnv(rollouts_itrs, env_itrs)
        else:
            pass

    def _plot_policies_PointEnv(self, rollouts_itrs, env_itrs):
        itr = 0
        rollouts_itrs = []
        final_rewards_itrs = []
        while os.path.exists(self._itr_file(itr)):
            # set seed
            if self.params['seed'] is not None:
                set_seed(self.params['seed'])

            tf_env = env_itrs[itr]
            env = tf_env
            while hasattr(env, 'wrapped_env'):
                env = env.wrapped_env
            xlim = [-1., 1.]
            ylim = [-1., 1.]
            xs = np.linspace(xlim[0], xlim[1], 10)
            ys = np.linspace(ylim[0], ylim[1], 10)

            sess, graph = MACPolicy.create_session_and_graph(gpu_device=self.params['policy']['gpu_device'],
                                                                   gpu_frac=self.params['policy']['gpu_frac'])
            with graph.as_default(), sess.as_default():
                policy = self._load_itr_policy(itr)

                final_rewards = np.inf * np.ones((len(xs), len(ys)), dtype=float)
                rollouts = []
                for i, x in enumerate(xs):
                    for j, y in enumerate(ys):
                        start_state = np.array([x, y])
                        tf_env.reset()
                        env.set_state(start_state)

                        rollout = rollout_policy(tf_env, policy, max_path_length=tf_env.horizon, start_obs=start_state)
                        rollouts.append(rollout)
                        final_rewards[i, j] = rollout['rewards'][-1]

            rollouts_itrs.append(rollouts)
            final_rewards_itrs.append(final_rewards)
            itr += 1

        ### normalize across all rewards
        for itr, final_rewards in enumerate(final_rewards_itrs):
            final_rewards_itrs[itr] = np.clip(final_rewards, -1., 0)

        ### plot
        f, axes = plt.subplots(1, len(final_rewards_itrs)+1, figsize=(10, 10))
        for itr, final_rewards in enumerate(final_rewards_itrs):
            ax = axes.ravel()[itr]
            ax.imshow(final_rewards, cmap='hot', vmin=-1., vmax=0., extent=xlim + ylim)
            ax.set_title('Itr {0:d}'.format(itr))

        matplotlib.colorbar.ColorbarBase(axes[-1], cmap='hot',
                                         norm=matplotlib.colors.Normalize(vmin=-1, vmax=0),
                                         orientation='vertical')
        for ax in axes.ravel():
            ax.set_aspect('equal')

        f.tight_layout()
        f.savefig(self._analyze_policy_img_file(None), bbox_inches='tight', dpi=200.)
        plt.close(f)

        # itr = 0
        # while os.path.exists(self._itr_file(itr)):
        #     N = 5
        #     f, axes = plt.subplots(N, N, figsize=(10, 10))
        #
        #     sess, graph = MACPolicy.create_session_and_graph()
        #     with graph.as_default(), sess.as_default():
        #         policy = self._load_itr_policy(itr)
        #
        #         observations = cartesian([np.linspace(l, u, N) for l, u in zip([-1., -1.], [1., 1.])])
        #         for ax, observation in zip(np.fliplr(axes.T).ravel(), observations):
        #             action, _ = policy.get_action(observation)
        #             ax.arrow(observation[0], observation[1], action[0], action[1], head_width=0.1, color='k')
        #             ax.plot([0], [0], color='r', marker='x', markersize=3.)
        #             ax.set_xlim((-2, 2))
        #             ax.set_ylim((-2, 2))
        #
        #     f.suptitle('Itr {0:d}'.format(itr))
        #     f.savefig(self._analyze_policy_img_file(itr), bbox_inches='tight', dpi=200.)
        #     plt.close(f)
        #
        #     itr += 1
        #     policy.terminate()

    def _plot_value_function(self, env_itrs):
        env = env_itrs[0]
        while hasattr(env, 'wrapped_env'):
            env = env.wrapped_env
        if isinstance(env, PointEnv):
            self._plot_value_function_PointEnv(env_itrs)
        else:
            pass

    def _plot_value_function_PointEnv(self, env_itrs):
        ### load policies and comptue values over grid
        values_itrs = []
        itr = 0
        xlim = [-1.5, 1.5]
        ylim = [-1.5, 1.5]
        N = 100
        xs = np.linspace(xlim[0], xlim[1], N)
        ys = np.linspace(ylim[0], ylim[1], N)
        while os.path.exists(self._itr_file(itr)):
            sess, graph = MACPolicy.create_session_and_graph(gpu_device=self.params['policy']['gpu_device'],
                                                                   gpu_frac=self.params['policy']['gpu_frac'])
            with graph.as_default(), sess.as_default():
                policy = self._load_itr_policy(itr)

                ### evaluate value over grid
                values = np.inf * np.ones((N, N), dtype=float)
                for i, x in enumerate(xs):
                    for j, y in enumerate(ys):
                        observation = np.array([x, y])
                        action, action_info = policy.get_action(observation, return_action_info=True)
                        values[i, j] = np.max(action_info['values'][0])

            values_itrs.append(values)
            itr += 1

        min_value = np.min(values_itrs)
        max_value = np.max(values_itrs)

        ### plot
        f, axes = plt.subplots(1, len(values_itrs) + 1, figsize=(10, 10))
        for itr, values in enumerate(values_itrs):
            ax = axes.ravel()[itr]
            ax.imshow(values, cmap='hot', extent=xlim+ylim, vmin=min_value, vmax=max_value)
            x_idx, y_idx = np.unravel_index(values.argmax(), values.shape)
            # ax.set_title('Itr {0:d}'.format(itr))
            ax.set_title('({0:.2f},{1:.2f})'.format(xs[x_idx], ys[y_idx]), {'fontsize': 8})

        matplotlib.colorbar.ColorbarBase(axes[-1], cmap='hot',
                                         norm=matplotlib.colors.Normalize(vmin=min_value, vmax=max_value),
                                         orientation='vertical')
        for ax in axes.ravel():
            ax.set_aspect('equal')

        f.tight_layout()
        f.savefig(self._analyze_value_function_img_file, bbox_inches='tight', dpi=200.)
        plt.close(f)

    def _plot_Q_function(self, env_itrs):
        env = env_itrs[0]
        while hasattr(env, 'wrapped_env'):
            env = env.wrapped_env
        if isinstance(env, PointEnv):
            self._plot_Q_function_PointEnv(env_itrs)
        else:
            pass

    def _plot_Q_function_PointEnv(self, env_itrs):
        ### get original env so can set state
        tf_env = env_itrs[0]
        env = tf_env
        while hasattr(env, 'wrapped_env'):
            env = env.wrapped_env

        N_start = 9
        xgrid, ygrid = np.meshgrid(np.linspace(-1., 1., N_start), np.linspace(-1., 1., N_start))
        start_states = [(x, y) for x, y in zip(xgrid.ravel(), ygrid.ravel())]

        N_action = 10
        axgrid, aygrid = np.meshgrid(np.linspace(-1., 1., N_action), np.linspace(-1., 1., N_action))
        actions = np.array([(ax, ay) for ax, ay in zip(axgrid.ravel(), aygrid.ravel())])

        next_states = [] # index by [state][action]
        for start_state in start_states:
            ### get next states
            next_states_i = []
            for action in actions:
                tf_env.reset()
                env.set_state(start_state)
                next_state, _, _, _ = tf_env.step(action)
                next_states_i.append(next_state)

            next_states.append(np.array(next_states_i))

        q_values = []
        itr = 0
        while os.path.exists(self._itr_file(itr)):
            sess, graph = MACPolicy.create_session_and_graph(gpu_device=self.params['policy']['gpu_device'],
                                                                   gpu_frac=self.params['policy']['gpu_frac'])
            with graph.as_default(), sess.as_default():
                policy = self._load_itr_policy(itr)

                q_values_itr = []
                for start_state in start_states:
                    q_values_itr.append(policy.eval_Q_values([start_state] * len(actions), actions))
                q_values.append(q_values_itr)

            itr += 1

        for itr, q_values_itr in enumerate(q_values):
            # N = int(np.ceil(np.sqrt(len(q_values_itr))))
            # f, axes = plt.subplots(N, N, figsize=(10, 10))
            f, ax = plt.subplots(1, 1, figsize=(10, 10))

            for i, (start_state, next_states_i, q_values_i) in enumerate(zip(start_states, next_states, q_values_itr)):
                # ax = np.flipud(axes).ravel()[i]
                ax.set_axis_bgcolor(matplotlib.cm.Greys(0.5))

                q_min = np.min(q_values_i)
                q_max = np.max(q_values_i)
                q_values_norm_i = (q_values_i - q_min) / (q_max - q_min)
                colors = [matplotlib.cm.plasma(q) for q in q_values_norm_i]

                ax.scatter(next_states_i[:, 0], next_states_i[:, 1], s=20. / float(N_action), c=colors)
                ax.plot([start_state[0]], [start_state[1]], 'kx', markersize=20. / float(N_action))
                ax.plot([next_states_i[q_values_i.argmax(), 0]], [next_states_i[q_values_i.argmax(), 1]],
                        color=colors[q_values_i.argmax()], marker='D', markersize=20. / float(N_action))

            # for ax in axes.ravel():
            ax.set_xlim((np.min(next_states) - 0.1, np.max(next_states) + 0.1))
            ax.set_ylim((np.min(next_states) - 0.1, np.max(next_states) + 0.1))

            suptitle = f.suptitle('Itr {0}'.format(itr), y=1.02)
            f.tight_layout()
            f.savefig(self._analyze_Q_function_img_file(itr), bbox_inches='tight', dpi=200.,
                      bbox_extra_artists=(suptitle,))
            plt.close(f)

    ###########
    ### Run ###
    ###########

    def run(self):
        logger.log('AnalyzeRNNCritic: plot_analyze')
        self._plot_analyze(self.train_rollouts_itrs, self.eval_rollouts_itrs)
        # logger.log('AnalyzeRNNCritic: plot_rollouts')
        # self._plot_rollouts(self.train_rollouts_itrs, self.eval_rollouts_itrs,
        #                     is_train=False, plot_prior=False)
        # self._plot_rollouts(self.train_rollouts_itrs, self.eval_rollouts_itrs,
        #                     is_train=True, plot_prior=False)
        # self._plot_policies(self.train_rollouts_itrs, self.env_itrs)
        # self._plot_value_function(self.env_itrs)
        # self._plot_Q_function(self.env_itrs)


def main(folder, skip_itr, max_itr):
    analyze = AnalyzeRNNCritic(os.path.join('/home/gkahn/code/rllab/data/local/rnn-critic/', folder),
                               skip_itr=skip_itr,
                               max_itr=max_itr)
    analyze.run()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('folder', type=str)
    parser.add_argument('--skip_itr', type=int, default=1)
    parser.add_argument('--max_itr', type=int, default=sys.maxsize)
    args = parser.parse_args()

    main(args.folder, args.skip_itr, args.max_itr)
