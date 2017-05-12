import os
import numpy as np
import itertools

import matplotlib.pyplot as plt
from matplotlib import ticker

from sandbox.gkahn.rnn_critic.examples.analyze_experiment import AnalyzeRNNCritic
from sandbox.gkahn.rnn_critic.utils.utils import DataAverageInterpolation

EXP_FOLDER = '/media/gkahn/ExtraDrive1/rllab/rnn_critic/'
SAVE_FOLDER = '/media/gkahn/ExtraDrive1/rllab/rnn_critic/final_plots'

########################
### Load experiments ###
########################

def load_experiments(indices):
    exps = []
    for i in indices:
        exps.append(AnalyzeRNNCritic(os.path.join(EXP_FOLDER, 'imswingup{0:03d}'.format(i)),
                                     clear_obs=False,
                                     create_new_envs=False))
    return exps

dqn_1 = load_experiments([0, 4, 5])
dqn_5 = load_experiments([6, 7, 8])
dqn_10 = load_experiments([16, 17, 18])
predictron_5 = load_experiments([9, 10, 11])
mac_5 = load_experiments([1, 2, 3])
mac_10 = load_experiments([12, 13, 14])

comparison_exps = np.array([[dqn_1, dqn_5, dqn_10, predictron_5, mac_5, mac_10]])

############
### Plot ###
############

def plot_cumreward(ax, analyze_group, color='k', label=None):
    data_interp = DataAverageInterpolation()
    min_step = max_step = None
    for analyze in analyze_group:
        rollouts = list(itertools.chain(*analyze.eval_rollouts_itrs))
        rollouts = sorted(rollouts, key=lambda r: r['steps'][0])
        steps = [r['steps'][0] for r in rollouts]
        values = [np.sum(r['rewards']) for r in rollouts]

        def moving_avg_std(idxs, data, window):
            avg_idxs, means, stds = [], [], []
            for i in range(window, len(data)):
                avg_idxs.append(np.mean(idxs[i - window:i]))
                means.append(np.mean(data[i - window:i]))
                stds.append(np.std(data[i - window:i]))
            return avg_idxs, np.asarray(means), np.asarray(stds)

        steps, values, _ = moving_avg_std(steps, values, window=1000)

        data_interp.add_data(steps, values)
        if min_step is None:
            min_step = steps[0]
        if max_step is None:
            max_step = steps[-1]
        min_step = max(min_step, steps[0])
        max_step = min(max_step, steps[-1])

    min_step = max(min_step, analyze.params['alg']['learn_after_n_steps'])

    steps = np.r_[min_step:max_step:500.][1:-1]
    values_mean, values_std = data_interp.eval(steps)
    steps -= min_step

    ax.plot(steps, values_mean, color=color, label=label)
    ax.fill_between(steps, values_mean - values_std, values_mean + values_std,
                    color=color, alpha=0.4)

    ax.grid()
    xfmt = ticker.ScalarFormatter()
    xfmt.set_powerlimits((0, 0))
    ax.xaxis.set_major_formatter(xfmt)

def plot_finalreward(ax, analyze_group, color='k', label=None, window=200):
    for i, analyze in enumerate(analyze_group):
        rollouts = list(itertools.chain(*analyze.eval_rollouts_itrs))
        rollouts = sorted(rollouts, key=lambda r: r['steps'][-1])
        steps = np.asarray([r['steps'][-1] for r in rollouts])
        values = np.asarray([r['rewards'][-1] for r in rollouts])

        def moving_avg_std(idxs, data, window):
            avg_idxs, means, stds = [], [], []
            for i in range(window, len(data)):
                avg_idxs.append(np.mean(idxs[i - window:i]))
                means.append(np.mean(data[i - window:i]))
                stds.append(np.std(data[i - window:i]))
            return avg_idxs, np.asarray(means), np.asarray(stds)

        def moving_median_std(idxs, data, window):
            median_idxs, medians, stds = [], [], []
            for i in range(window, len(data)):
                median_idxs.append(np.mean(idxs[i - window:i]))
                medians.append(np.median(data[i - window:i]))
                stds.append(np.std(data[i - window:i]))
            return median_idxs, np.asarray(medians), np.asarray(stds)

        # values[values < -np.pi] = -np.pi
        # steps, values, values_std = moving_avg_std(steps, values, window=window)
        steps, values, values_std = moving_median_std(steps, values, window=window)
        values[values < -np.pi] = -np.pi

        ax.plot(steps, values, color=['r', 'g', 'b'][i])
        # ax.fill_between(steps, values_mean - values_std, values_mean + values_std,
        #                 color=color, alpha=0.4)

        threshold = -np.sqrt(5. * np.pi / 180.)
        ax.hlines(threshold, steps[0], steps[-1], color='k', linestyle='--')
        print(steps[(values > threshold).argmax()])
    print('')

    ax.grid()
    xfmt = ticker.ScalarFormatter()
    xfmt.set_powerlimits((0, 0))
    ax.xaxis.set_major_formatter(xfmt)

shape = comparison_exps.shape[:2]
f, axes = plt.subplots(*shape, figsize=(15, 3), sharex=True, sharey=True)
axes = axes.reshape(shape)
for i in range(shape[0]):
    for j in range(shape[1]):
        ax = axes[i, j]
        # plot_cumreward(ax, comparison_exps[i, j, :])
        plot_finalreward(ax, comparison_exps[i, j, :], window=100) # 500

for i, name in enumerate(['DQN', 'DQN-5', 'DQN-10', 'Predictron-5', 'MAC-5', 'MAC-10']):
    axes[0, i].set_title(name)

# plt.tight_layout()
f.savefig(os.path.join(SAVE_FOLDER, 'imswingup_comparison.png'), bbox_inches='tight', dpi=200)
plt.close(f)

import IPython; IPython.embed()
