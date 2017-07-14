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
            exps.append(AnalyzeRNNCritic(os.path.join(EXP_FOLDER, 'walker{0:03d}'.format(i)),
                                         clear_obs=False,
                                         create_new_envs=False,
                                         load_train_rollouts=False,
                                         load_eval_rollouts=False))
            print(i)
        except:
            pass

    return exps

all_exps = [load_experiments(range(i, i+3)) for i in list(range(10, 24, 3))]

############
### Plot ###
############

def plot_cumreward(ax, analyze_group, color='k', label=None, window=20):
    data_interp = DataAverageInterpolation()
    min_step = max_step = None
    for i, analyze in enumerate(analyze_group):
        steps = np.array(analyze.progress['Step'])
        values = np.array(analyze.progress['EvalCumRewardMean'])

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

# for i, exps in enumerate(list(np.array_split(all_exps[:-(len(all_exps) % 24)], len(all_exps) // 24)) + [all_exps[-(len(all_exps) % 24):]]):
exps = all_exps
f_cumreward, axes_cumreward = plt.subplots(1, 5, figsize=(15, 3), sharey=True, sharex=True)

for ax_cumreward, exp in zip(axes_cumreward.ravel(), exps):
    if not hasattr(exp, '__len__'):
        exp = [exp]

    if len(exp) > 0:
        # try:
        plot_cumreward(ax_cumreward, exp, window=20)
        params = exp[0].params
        policy = params['policy'][params['policy']['class']]
        ax_cumreward.set_title('N: {0}, H: {1}'.format(
            params['policy']['N'],
            params['policy']['H'],
        ), fontdict={'fontsize': 8})
        # except:
        #     print('Could not plot exp')

f_cumreward.savefig(os.path.join(SAVE_FOLDER, 'walker0_comparison.png'), bbox_inches='tight', dpi=150)
plt.close(f_cumreward)
