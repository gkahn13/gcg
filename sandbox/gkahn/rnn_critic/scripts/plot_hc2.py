import os
import numpy as np
import itertools

import matplotlib.pyplot as plt
from matplotlib import ticker
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from matplotlib.gridspec import GridSpec

from analyze_experiment import AnalyzeRNNCritic
from sandbox.gkahn.rnn_critic.utils.utils import DataAverageInterpolation

EXP_FOLDERS = ['/media/gkahn/ExtraDrive1/rllab/rnn_critic/', '/media/gkahn/ExtraDrive1/rllab/s3/rnn-critic']
SAVE_FOLDER = '/media/gkahn/ExtraDrive1/rllab/rnn_critic/final_plots'

########################
### Load experiments ###
########################

def load_experiments(indices, plot_dict):
    exps = []
    for i in indices:
        for EXP_FOLDER in EXP_FOLDERS:
            try:
                analyze = AnalyzeRNNCritic(os.path.join(EXP_FOLDER, 'hc{0:03d}'.format(i)),
                                             clear_obs=True,
                                             create_new_envs=False,
                                             load_train_rollouts=False,
                                             load_eval_rollouts=False)
                # rollouts = list(itertools.chain(*analyze.eval_rollouts_itrs))
                # rollouts = sorted(rollouts, key=lambda r: r['steps'][0])
                # steps = [r['steps'][0] for r in rollouts]
                # values = [np.sum(r['rewards']) for r in rollouts]

                indices = np.nonzero(analyze.progress['EvalNumEpisodes'] > 0)[0]
                steps = np.array(analyze.progress['Step'][indices]).ravel()
                values = np.array(analyze.progress['EvalCumRewardMean'][indices]).ravel()

                exps.append((steps, values, plot_dict))
                print(i)
                break
            except:
                pass

    return exps

dqn_5_wb_mean = load_experiments([56, 57, 58], {'color': 'g', 'label': r'mean', 'linestyle': '-'})
dqn_5_wb_final = load_experiments([91, 92, 93], {'color': 'g', 'label': r'final', 'linestyle': '-.'})
dqn_5_wb_exp_25 = load_experiments([94, 95, 96], {'color': 'g', 'label': r'exponential', 'linestyle': ':'})
dqn_5_wb_exp_50 = load_experiments([97, 98, 99], {'color': 'g', 'label': r'exponential', 'linestyle': ':'})
dqn_5_wb_exp_75 = load_experiments([100, 101, 102], {'color': 'g', 'label': r'exponential', 'linestyle': ':'})
dqn_5_wb_exp_95 = load_experiments([103, 104, 105], {'color': 'g', 'label': r'exponential', 'linestyle': ':'})
dqn_5_wb_exps = [dqn_5_wb_exp_25, dqn_5_wb_exp_50, dqn_5_wb_exp_75, dqn_5_wb_exp_95]
dqn_5_wb_exp = dqn_5_wb_exps[np.argmax([np.mean([exp[1][-1] for exp in exps]) for exps in dqn_5_wb_exps])]
mac_5_wb_mean = load_experiments([48, 49], {'color': 'b', 'label': r'MAC', 'linestyle': '-'})

dqn_10_wb_mean = load_experiments([59, 60, 61], {'color': 'g', 'label': r'mean', 'linestyle': '-'})
dqn_10_wb_final = load_experiments([121, 122, 123], {'color': 'g', 'label': r'final', 'linestyle': '-.'})
dqn_10_wb_exp_25 = load_experiments([124, 125, 126], {'color': 'g', 'label': r'exponential', 'linestyle': ':'})
dqn_10_wb_exp_50 = load_experiments([127, 128, 129], {'color': 'g', 'label': r'exponential', 'linestyle': ':'})
dqn_10_wb_exp_75 = load_experiments([130, 131, 132], {'color': 'g', 'label': r'exponential', 'linestyle': ':'})
dqn_10_wb_exp_95 = load_experiments([133, 134, 135], {'color': 'g', 'label': r'exponential', 'linestyle': ':'})
dqn_10_wb_exps = [dqn_10_wb_exp_25, dqn_10_wb_exp_50, dqn_10_wb_exp_75, dqn_10_wb_exp_95]
dqn_10_wb_exp = dqn_10_wb_exps[np.argmax([np.mean([exp[1][-1] for exp in exps]) for exps in dqn_10_wb_exps])]
mac_10_wb_mean = load_experiments([50, 51], {'color': 'b', 'label': r'MAC', 'linestyle': '-'})


### what we will use
dqn_1 = load_experiments([53, 54, 55], {})
dqn_5 = dqn_5_wb_final
mac_5 = mac_5_wb_mean
dqn_10 = dqn_10_wb_exp
mac_10 = mac_10_wb_mean

import IPython; IPython.embed()

############
### Plot ###
############

def plot_cumreward(ax, analyze_group, color='k', label=None, window=25):
    data_interp = DataAverageInterpolation()
    min_step = max_step = None
    for i, (steps, values, plot) in enumerate(analyze_group):

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

    # max_step = min(2e6, max_step)

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

plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=18)

f, axes = plt.subplots(1, 2, figsize=(8, 3), sharex=True, sharey=True)

plot_cumreward(axes[0], dqn_1, color='k')
plot_cumreward(axes[0], dqn_5, color='g')
plot_cumreward(axes[0], mac_5, color='b')

plot_cumreward(axes[1], dqn_1, color='k', label='Double Q-learning')
plot_cumreward(axes[1], dqn_10, color='g', label='Multistep Double Q-learning')
plot_cumreward(axes[1], mac_10, color='b', label='MAQL (ours)')

# axes[1].legend(loc='center right', ncol=1, bbox_to_anchor=(2.3, 0.5))
axes[1].legend(loc='upper center', ncol=2, bbox_to_anchor=(-0.15, 1.55))

axes[0].set_ylabel(r'Cumulative reward')
for ax in axes.ravel():
    ax.set_xlabel('Steps')
axes[0].set_title(r'$N=5$')
axes[1].set_title(r'$N=10$')

f.savefig(os.path.join(SAVE_FOLDER, 'hc2_comparison.png'), bbox_inches='tight', dpi=200)
plt.close(f)


