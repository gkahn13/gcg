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

def load_experiments(indices, plot={}):
    exps = []
    for i in indices:
        try:
            exps.append(AnalyzeRNNCritic(os.path.join(EXP_FOLDER, 'walker{0:03d}'.format(i)),
                                         plot=plot,
                                         clear_obs=False,
                                         create_new_envs=False))
        except:
            pass
    return exps

dqn_5 = load_experiments([1, 2], plot={'color': 'k', 'label': 'DQN'})
dqn_10 = load_experiments([3, 4], plot={'color': 'k', 'label': 'DQN'})

mac_5 = load_experiments([5, 6], plot={'color': 'r', 'label': 'MAC'})
mac_10 = load_experiments([7, 8], plot={'color': 'r', 'label': 'MAC'})

comparison_exps = np.array([[dqn_5, dqn_10],
                            [mac_5, mac_10]])

############
### Plot ###
############

import IPython; IPython.embed()

def plot_cumreward(ax, analyze_group, window=100):
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

        steps, values, _ = moving_avg_std(steps, values, window=window)

        data_interp.add_data(steps, values)
        if min_step is None:
            min_step = steps[0]
        if max_step is None:
            max_step = steps[-1]
        min_step = max(min_step, steps[0])
        max_step = min(max_step, steps[-1])

    steps = np.r_[min_step:max_step:50.][1:-1]
    values_mean, values_std = data_interp.eval(steps)
    steps -= min_step

    ax.plot(steps, values_mean, color=analyze.plot['color'], label=analyze.plot['label'])
    ax.fill_between(steps, values_mean - values_std, values_mean + values_std,
                    color=analyze.plot['color'], alpha=0.4)

K = comparison_exps.shape[1]
f, axes = plt.subplots(1, K, figsize=(20, 6), sharex=True, sharey=True)
for j, N in enumerate([5, 10]):
    ax = axes.ravel()[j]
    for i in range(len(comparison_exps)):
        plot_cumreward(ax, comparison_exps[i, j], window=100)

    ax.set_title('N = {0}'.format(N))
    ax.legend(loc='upper left')
    ax.grid()
    xfmt = ticker.ScalarFormatter()
    xfmt.set_powerlimits((0, 0))
    ax.xaxis.set_major_formatter(xfmt)

# plt.tight_layout()
f.savefig(os.path.join(SAVE_FOLDER, 'walker_comparison.png'), bbox_inches='tight', dpi=200)
plt.close(f)


