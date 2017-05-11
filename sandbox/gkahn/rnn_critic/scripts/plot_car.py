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
        exps.append(AnalyzeRNNCritic(os.path.join(EXP_FOLDER, 'car{0:03d}'.format(i)),
                                     clear_obs=False,
                                     create_new_envs=False))
    return exps

dqn_1 = load_experiments([8, 9, 10])
mac_5_targ_1 = load_experiments([11, 12, 13])
mac_5_targ_5 = load_experiments([14, 15, 16])

comparison_exps = np.array([[dqn_1, mac_5_targ_1, mac_5_targ_5]])

############
### Plot ###
############

def plot_cumreward(ax, analyze_group, color='k', label=None, window=500):
    for i, analyze in enumerate(analyze_group):
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

        steps, values, _ = moving_avg_std(steps, values, window=window)

        ax.plot(steps, values, label=label, color=['r', 'g', 'b'][i])

    ax.grid()
    xfmt = ticker.ScalarFormatter()
    xfmt.set_powerlimits((0, 0))
    ax.xaxis.set_major_formatter(xfmt)

shape = comparison_exps.shape[:2]
f, axes = plt.subplots(*shape, figsize=(10, 3), sharex=True, sharey=True)
axes = axes.reshape(shape)
for i in range(shape[0]):
    for j in range(shape[1]):
        ax = axes[i, j]
        plot_cumreward(ax, comparison_exps[i, j, :], window=200)

for i, name in enumerate(['DQN', 'MAC-5-targ-1', 'MAC-5-targ-5']):
    axes[0, i].set_title(name)

# plt.tight_layout()
f.savefig(os.path.join(SAVE_FOLDER, 'car_comparison.png'), bbox_inches='tight', dpi=200)
plt.close(f)

import IPython; IPython.embed()
