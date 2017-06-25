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
            exps.append(AnalyzeRNNCritic(os.path.join(EXP_FOLDER, 'pong{0:03d}'.format(i)),
                                         clear_obs=False,
                                         create_new_envs=False,
                                         load_train_rollouts=False,
                                         load_eval_rollouts=False))
            print(i)
        except:
            pass

    return exps

all_exps = [load_experiments(range(i, i+3)) for i in list(range(214, 573, 3)) + \
                                                     [718, 721] + \
                                                     list(range(725, 796, 3)) + \
                                                     list(range(798, 821, 3))]

import IPython; IPython.embed()

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

for i, exps in enumerate(list(np.array_split(all_exps[:-(len(all_exps) % 25)], len(all_exps) // 25)) + [all_exps[-(len(all_exps) % 25):]]):
    f, axes = plt.subplots(5, 5, figsize=(15, 15), sharey=True, sharex=True)

    if not hasattr(axes, 'ravel'):
        axes = np.array([axes])

    for ax, exp in zip(axes.ravel(), exps):
        if not hasattr(exp, '__len__'):
            exp = [exp]

        if len(exp) > 0:
            try:
                plot_cumreward(ax, exp)
                params = exp[0].params
                policy = params['policy'][params['policy']['class']]
                # class, N, H, update target, K target, lstm, lr
                ax.set_title('{7}, {0}, N: {1}, H: {2}, target update: {3},\nK target: {4}, lstm: {5}, lr: {6}'.format(
                    params['policy']['class'],
                    params['policy']['N'],
                    params['policy']['H'],
                    params['alg']['update_target_every_n_steps'],
                    params['policy']['get_action_target']['random']['K'],
                    policy['use_lstm'] if 'use_lstm' in policy else False,
                    params['policy']['lr_schedule']['outside_value'],
                    params['exp_name']
                ), fontdict={'fontsize': 5})
            except:
                print('Could not plot exp')

        ax.set_ylim((-22, 22))
        ax.set_xlim((0, 1e6))

    f.savefig(os.path.join(SAVE_FOLDER, 'pong3_{0:02d}_comparison.png'.format(i)), bbox_inches='tight', dpi=300)
    plt.close(f)
