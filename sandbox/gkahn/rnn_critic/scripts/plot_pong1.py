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

dqn_1 = load_experiments([161, 162, 163])
dqn_5_wb = load_experiments([164, 165, 166])
dqn_5_et = load_experiments([167, 168, 169])
dqn_10_wb = load_experiments([176, 177, 178])
dqn_10_et = load_experiments([179, 180, 181])

import IPython; IPython.embed()

############
### Plot ###
############

def plot_cumreward(ax, analyze_group, color='k', label=None, window=50):
    data_interp = DataAverageInterpolation()
    min_step = max_step = None
    for i, analyze in enumerate(analyze_group):
        rollouts = list(itertools.chain(*analyze.eval_rollouts_itrs))
        rollouts = sorted(rollouts, key=lambda r: r['steps'][0])
        steps = [4 * r['steps'][0] for r in rollouts]
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

bg = [ 0.25882353,  0.95686275,  0.69019608]

f, axes = plt.subplots(1, 2, figsize=(6, 2), sharey=True, sharex=True)

plot_cumreward(axes[0], dqn_1, color='k', label=r'Standard critic')
plot_cumreward(axes[0], dqn_5_et, color=bg, label=r'MRC ET')
plot_cumreward(axes[0], dqn_5_wb, color='g', label=r'MRC CMR (ours)')

plot_cumreward(axes[1], dqn_1, color='k', label=r'Standard critic')
plot_cumreward(axes[1], dqn_10_et, color=bg, label=r'MRC ET')
plot_cumreward(axes[1], dqn_10_wb, color='g', label=r'MRC CMR (ours)')

ax = axes[0]
ax.set_ylabel('Episode Reward')
ax.legend(loc='upper center', bbox_to_anchor=(1.1, 1.25), ncol=3)
for ax in axes.ravel():
    ax.set_xlabel('Frames')

artists = [axes[0].text(0.5, -0.5, r'(i) $N=5$', ha='center', transform=axes[0].transAxes),
           axes[1].text(0.5, -0.5, r'(ii) $N=10$', ha='center', transform=axes[1].transAxes)]


# plt.tight_layout()
f.savefig(os.path.join(SAVE_FOLDER, 'pong1_comparison.png'), bbox_inches='tight', dpi=300)
plt.close(f)

