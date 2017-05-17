import os
import numpy as np
import itertools

import matplotlib.pyplot as plt
from matplotlib import ticker

from analyze_experiment import AnalyzeRNNCritic
from sandbox.gkahn.rnn_critic.utils.utils import DataAverageInterpolation

EXP_FOLDER = '/media/gkahn/ExtraDrive1/rllab/rnn_critic/'
SAVE_FOLDER = '/media/gkahn/ExtraDrive1/rllab/rnn_critic/final_plots'

########################
### Load experiments ###
########################

def load_experiments(indices):
    exps = []
    for i in indices:
        try:
            analyze = AnalyzeRNNCritic(os.path.join(EXP_FOLDER, 'tennis{0:03d}'.format(i)),
                                         clear_obs=True,
                                         create_new_envs=False,
                                         load_train_rollouts=False)
            rollouts = list(itertools.chain(*analyze.eval_rollouts_itrs))
            rollouts = sorted(rollouts, key=lambda r: r['steps'][0])
            steps = [r['steps'][0] for r in rollouts]
            values = [np.sum(r['rewards']) for r in rollouts]
            exps.append((steps, values))
            print(i)
        except:
            pass

    return exps

dqn_1 = load_experiments([31, 32, 33])
dqn_5_sep = load_experiments([34, 35, 36])
dqn_5 = load_experiments([37, 38, 39])
dqn_10_sep = load_experiments([40, 41, 42])
dqn_10 = load_experiments([43, 44, 45])

comparison_exps = np.array([[dqn_1, dqn_5, dqn_10],
                            [[], dqn_5_sep, dqn_10_sep]])

############
### Plot ###
############

def plot_cumreward(ax, analyze_group, color='k', label=None, window=100):
    data_interp = DataAverageInterpolation()
    min_step = max_step = None
    for i, (steps, values) in enumerate(analyze_group):

        def moving_avg_std(idxs, data, window):
            avg_idxs, means, stds = [], [], []
            for i in range(window, len(data)):
                avg_idxs.append(np.mean(idxs[i - window:i]))
                means.append(np.mean(data[i - window:i]))
                stds.append(np.std(data[i - window:i]))
            return avg_idxs, np.asarray(means), np.asarray(stds)

        steps, values, _ = moving_avg_std(steps, values, window=window)

        ax.plot(steps, values, color='k', alpha=np.linspace(1., 0.4, len(analyze_group))[i])

        if len(steps) == 0:
            continue

        data_interp.add_data(steps, values)
        if min_step is None:
            min_step = steps[0]
        if max_step is None:
            max_step = steps[-1]
        min_step = max(min_step, steps[0])
        max_step = min(max_step, steps[-1])

    if len(analyze_group) > 1:
        steps = np.r_[min_step:max_step:50.][1:-1]
        values_mean, values_std = data_interp.eval(steps)
        # steps -= min_step

        ax.plot(steps, values_mean, color='r', label=label)
        ax.fill_between(steps, values_mean - values_std, values_mean + values_std,
                        color='r', alpha=0.4)

    ax.grid()
    xfmt = ticker.ScalarFormatter()
    xfmt.set_powerlimits((0, 0))
    ax.xaxis.set_major_formatter(xfmt)

import IPython; IPython.embed()

shape = comparison_exps.shape[:2]
f, axes = plt.subplots(*shape, figsize=(15, 5), sharex=True, sharey=True)
for i in range(shape[0]):
    for j in range(shape[1]):
        print(i, j)
        ax = axes[i, j]
        exp = comparison_exps[i, j]
        if len(exp) > 0:
            plot_cumreward(ax, exp, window=30)

for i, name in enumerate(['DQN', 'DQN sep']):
    axes[i, 0].set_ylabel(name)
for j, N in enumerate([1, 5, 10]):
    axes[0, j].set_title(str(N))

# plt.tight_layout()
f.savefig(os.path.join(SAVE_FOLDER, 'tennis_comparison.png'), bbox_inches='tight', dpi=200)
plt.close(f)

import IPython; IPython.embed()
