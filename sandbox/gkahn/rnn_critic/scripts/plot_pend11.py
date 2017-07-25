import os
import numpy as np
import itertools

import matplotlib.pyplot as plt
from matplotlib import ticker

from analyze_experiment import AnalyzeRNNCritic
from sandbox.gkahn.rnn_critic.utils.utils import DataAverageInterpolation

EXP_FOLDER = '/media/gkahn/ExtraDrive1/rllab/s3/rnn-critic/'
SAVE_FOLDER = '/media/gkahn/ExtraDrive1/rllab/rnn_critic/final_plots'

########################
### Load experiments ###
########################

def load_experiments(indices):
    exps = []
    for i in indices:
        try:
            exps.append(AnalyzeRNNCritic(os.path.join(EXP_FOLDER, 'pend{0:03d}'.format(i)),
                                         clear_obs=False,
                                         create_new_envs=False,
                                         load_train_rollouts=False))
            print(i)
        except:
            pass

    return exps

exps_list = [load_experiments(range(i, i+3)) for i in list(range(1311, 1340, 3))]

import IPython; IPython.embed()

############
### Plot ###
############

def plot_cumreward(ax, analyze_group, color='k', label=None, window=20):
    data_interp = DataAverageInterpolation()
    min_step = max_step = None
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

        # ax.plot(steps, values, color='k', alpha=np.linspace(1., 0.4, len(analyze_group))[i])

        data_interp.add_data(steps, values)
        if min_step is None:
            min_step = steps[0]
        if max_step is None:
            max_step = steps[-1]
        min_step = max(min_step, steps[0])
        max_step = min(max_step, steps[-1])

    steps = np.r_[min_step:max_step:50.][1:-1]
    values_mean, values_std = data_interp.eval(steps)
    # steps -= min_step

    ax.plot(steps, values_mean, color=color, label=label)
    ax.fill_between(steps, values_mean - values_std, values_mean + values_std,
                    color=color, alpha=0.4)

    ax.grid()
    xfmt = ticker.ScalarFormatter()
    xfmt.set_powerlimits((0, 0))
    ax.xaxis.set_major_formatter(xfmt)

nrows = ncols = int(np.ceil(np.sqrt(len(exps_list))))

f, axes = plt.subplots(nrows, ncols, figsize=(15, 15), sharex=True, sharey=True)

for ax, exps in zip(axes.ravel(), exps_list):
    if len(exps) > 0:
        plot_cumreward(ax, exps)
        params = exps[0].params
        ax.set_title('{0}, N: {1}, H: {2}'.format(
            params['policy']['class'],
            params['policy']['N'],
            params['policy']['H']))

# plt.tight_layout()
f.savefig(os.path.join(SAVE_FOLDER, 'pend11_comparison.png'), bbox_inches='tight', dpi=150)
plt.close(f)


