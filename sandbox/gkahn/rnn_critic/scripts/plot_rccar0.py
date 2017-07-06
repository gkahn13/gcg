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
            exps.append(AnalyzeRNNCritic(os.path.join(EXP_FOLDER, 'rccar{0:03d}'.format(i)),
                                         clear_obs=False,
                                         create_new_envs=False,
                                         load_train_rollouts=True,
                                         load_eval_rollouts=True))
            print(i)
        except:
            pass

    return exps

all_exps = [load_experiments(range(i, i+3)) for i in [0]]

############
### Plot ###
############

# distance travelled
# time between crashes

def plot_distance(ax, analyze_group, color='k', label=None, window=100):
    data_interp = DataAverageInterpolation()
    min_step = max_step = None
    for i, analyze in enumerate(analyze_group):
        steps = np.array([r['steps'][0] for r in itertools.chain(*analyze.train_rollouts_itrs)])
        values = np.array([np.sum(0.25 * 8. * r['actions'][:,1]) for r in itertools.chain(*analyze.train_rollouts_itrs)])
       
        def moving_avg_std(idxs, data, window):
            avg_idxs, means, stds = [], [], []
            for i in range(window, len(data)):
                avg_idxs.append(np.mean(idxs[i - window:i]))
                means.append(np.mean(data[i - window:i]))
                stds.append(np.std(data[i - window:i]))
            return avg_idxs, np.asarray(means), np.asarray(stds)

        steps, values, _ = moving_avg_std(steps, values, window=window)

        ax.plot(steps, values, color='r', alpha=np.linspace(1., 0.4, len(analyze_group))[i])

        try:
            data_interp.add_data(steps, values)
        except:
            continue
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

f, axes = plt.subplots(len(all_exps), 1, figsize=(10, 10), sharey=True, sharex=True)
if len(all_exps) <= 1:
    axes = np.array([axes])
for ax, exps in zip(axes.ravel(), all_exps):
    plot_distance(ax, exps)
f.savefig(os.path.join(SAVE_FOLDER, 'rccar0_distance.png'), bbox_inches='tight', dpi=150)
