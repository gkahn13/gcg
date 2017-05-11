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
        exps.append(AnalyzeRNNCritic(os.path.join(EXP_FOLDER, 'hc{0:03d}'.format(i)),
                                     clear_obs=False,
                                     create_new_envs=False))
    return exps

mac_1 = load_experiments([42, 43, 44])
mac_5_target_1 = load_experiments([45, 46, 47])
mac_5_target_5 = load_experiments([48, 49])
mac_10_target_10 = load_experiments([50, 51])

comparison_exps = np.array([[mac_1, mac_5_target_1, mac_5_target_5, mac_10_target_10]])

############
### Plot ###
############

def plot_cumreward(ax, analyze_group, color='k', label=None, window=100):
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

        steps, values, values_std = moving_avg_std(steps, values, window=window)

        ax.plot(steps, values, color=color)
        # ax.fill_between(steps, values - values_std, values + values_std, color=color, alpha=0.4)

    ax.grid()
    xfmt = ticker.ScalarFormatter()
    xfmt.set_powerlimits((0, 0))
    ax.xaxis.set_major_formatter(xfmt)

import IPython; IPython.embed()

shape = comparison_exps.shape[:2]
f, axes = plt.subplots(*shape, figsize=(10, 3), sharex=True, sharey=True)
axes = axes.reshape(shape)
for i in range(shape[0]):
    for j in range(shape[1]):
        ax = axes[i, j]
        plot_cumreward(ax, comparison_exps[i, j])

for i, name in enumerate(['MAC-1', 'MAC-5-targ-1', 'MAC-5-targ-5', 'MAC-10-targ-10']):
    axes[0, i].set_title(name)

# plt.tight_layout()
f.savefig(os.path.join(SAVE_FOLDER, 'hc_comparison.png'), bbox_inches='tight', dpi=200)
plt.close(f)


