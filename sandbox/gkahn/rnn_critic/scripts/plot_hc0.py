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

            exps.append((steps, values))
            print(i)
        except:
            pass

    return exps

dqn_1 = load_experiments([53, 54, 55])
mac_1 = load_experiments([42, 43, 44])

dqn_5_wb = load_experiments([56, 57, 58])
dqn_5_et = load_experiments([70, 71, 72])
mac_5_wb = load_experiments([48, 49])
mac_5_et = load_experiments([74, 75, 76])

dqn_10_wb = load_experiments([59, 60, 61])
dqn_10_et = load_experiments([77, 78, 79])
mac_10_wb = load_experiments([50, 51])
mac_10_et = load_experiments([80, 81, 82])

############
### Plot ###
############

import IPython; IPython.embed()

def plot_cumreward(ax, analyze_group, color='k', label=None, window=50):
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

        # ax.plot(steps, values, color='k', alpha=np.linspace(1., 0.4, len(analyze_group))[i])

        data_interp.add_data(steps, values)
        if min_step is None:
            min_step = steps[0]
        if max_step is None:
            max_step = steps[-1]
        min_step = max(min_step, steps[0])
        max_step = min(max_step, steps[-1])

    max_step = min(1.5e6, max_step)

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

f, axes = plt.subplots(1, 3, figsize=(9, 2), sharey=True, sharex=True)

plot_cumreward(axes[0], dqn_1, color='k', label=r'$Q$')
plot_cumreward(axes[0], mac_1, color='r', label=r'MAC')

plot_cumreward(axes[1], dqn_5_et, color='c', label='$Q_N$, eligibility trace')
plot_cumreward(axes[1], dqn_5_wb, color='b', label='$Q_N$, weighted Bellman')
plot_cumreward(axes[1], mac_5_et, color='m', label='MAC, eligibility trace')
plot_cumreward(axes[1], mac_5_wb, color='r', label='MAC, weighted Bellman')

plot_cumreward(axes[2], dqn_10_et, color='c')
plot_cumreward(axes[2], dqn_10_wb, color='b')
plot_cumreward(axes[2], mac_10_et, color='m')
plot_cumreward(axes[2], mac_10_wb, color='r')

for ax in axes.ravel():
    ax.grid()
axes[0].legend(loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.4))
axes[1].legend(loc='upper right', ncol=2, bbox_to_anchor=(2.2, 1.55))

axes[0].set_title(r'$N = 1$')
axes[1].set_title(r'$N = 5$')
axes[2].set_title(r'$N = 10$')

for ax in axes.ravel():
    ax.set_xlabel('Steps')
axes[0].set_ylabel('Episode Reward')

# plt.tight_layout()
f.savefig(os.path.join(SAVE_FOLDER, 'hc0_comparison.png'), bbox_inches='tight', dpi=300)
plt.close(f)


