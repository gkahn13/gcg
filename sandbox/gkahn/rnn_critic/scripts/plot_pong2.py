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
            exps.append(AnalyzeRNNCritic(os.path.join(EXP_FOLDER, 'pong{0:03d}'.format(i)),
                                         clear_obs=False,
                                         create_new_envs=False,
                                         load_train_rollouts=False))
            print(i)
        except:
            pass

    return exps

dqn = load_experiments([189, 190, 191])

mac_5_lattice = load_experiments([210, 211, 212])
mac_5 = load_experiments([192, 193, 194])
mac_10 = load_experiments([195, 196, 197])
mac_20 = load_experiments([198, 199, 200])

rand_mac_5 = load_experiments([201, 202, 203])
rand_mac_10 = load_experiments([204, 205, 206])
rand_mac_20 = load_experiments([207, 208, 209])

import IPython; IPython.embed()

############
### Plot ###
############

def plot_cumreward(ax, analyze_group, color='k', label=None, window=50):
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

        ax.plot(steps, values, color='k', alpha=np.linspace(1., 0.4, len(analyze_group))[i])

    ax.grid()
    xfmt = ticker.ScalarFormatter()
    xfmt.set_powerlimits((0, 0))
    ax.xaxis.set_major_formatter(xfmt)

f, axes = plt.subplots(3, 5, figsize=(10, 6), sharey=True, sharex=True)

plot_cumreward(axes[0, 0], dqn)

plot_cumreward(axes[1, 1], mac_5_lattice)
plot_cumreward(axes[1, 2], mac_5)
plot_cumreward(axes[1, 3], mac_10)
plot_cumreward(axes[1, 4], mac_20)

plot_cumreward(axes[2, 2], rand_mac_5)
plot_cumreward(axes[2, 3], rand_mac_10)
plot_cumreward(axes[2, 4], rand_mac_20)

axes[0, 0].set_ylabel('DQN')
axes[1, 0].set_ylabel('MAQL')
axes[2, 0].set_ylabel('Random MAQL')

axes[0, 0].set_title('N=1')
axes[0, 1].set_title('N=5 (lattice)')
axes[0, 2].set_title('N=5')
axes[0, 3].set_title('N=10')
axes[0, 4].set_title('N=20')

# plt.tight_layout()
f.savefig(os.path.join(SAVE_FOLDER, 'pong2_comparison.png'), bbox_inches='tight', dpi=300)
plt.close(f)

