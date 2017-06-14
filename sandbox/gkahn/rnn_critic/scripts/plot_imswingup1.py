import os
import numpy as np
import itertools

import matplotlib.pyplot as plt
from matplotlib import ticker
import matplotlib.patches as mpatches

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
        print('Loading {0}'.format(i))
        exps.append(AnalyzeRNNCritic(os.path.join(EXP_FOLDER, 'imswingup{0:03d}'.format(i)),
                                     clear_obs=False,
                                     create_new_envs=False,
                                     load_train_rollouts=False))
    return exps

dqn_1 = load_experiments([0, 4, 5])
dqn_5 = load_experiments([6, 7, 8])
dqn_10 = load_experiments([16, 17, 18])
mac_5 = load_experiments([1, 2, 3])
mac_10 = load_experiments([12, 13, 14])

############
### Plot ###
############

def get_threshold(analyze_group, window=200):
    steps_threshold = []
    for i, analyze in enumerate(analyze_group):
        rollouts = list(itertools.chain(*analyze.eval_rollouts_itrs))
        rollouts = sorted(rollouts, key=lambda r: r['steps'][-1])
        steps = np.asarray([r['steps'][-1] for r in rollouts])
        values = np.asarray([r['rewards'][-1] for r in rollouts])

        def moving_avg_std(idxs, data, window):
            avg_idxs, means, stds = [], [], []
            for i in range(window, len(data)):
                avg_idxs.append(np.mean(idxs[i - window:i]))
                means.append(np.mean(data[i - window:i]))
                stds.append(np.std(data[i - window:i]))
            return avg_idxs, np.asarray(means), np.asarray(stds)

        def moving_median_std(idxs, data, window):
            median_idxs, medians, stds = [], [], []
            for i in range(window, len(data)):
                median_idxs.append(np.mean(idxs[i - window:i]))
                medians.append(np.median(data[i - window:i]))
                stds.append(np.std(data[i - window:i]))
            return median_idxs, np.asarray(medians), np.asarray(stds)

        # values[values < -np.pi] = -np.pi
        # steps, values, values_std = moving_avg_std(steps, values, window=window)
        steps, values, values_std = moving_median_std(steps, values, window=window)
        values[values < -np.pi] = -np.pi

        threshold = -np.sqrt(5. * np.pi / 180.)
        if (values > threshold).max() > 0:
            steps_threshold.append(steps[(values > threshold).argmax()])
        else:
            steps_threshold.append(steps[-1])

    return np.mean(steps_threshold), np.std(steps_threshold)

import IPython; IPython.embed()

comparison_exps = [dqn_1, dqn_5, mac_5, dqn_10, mac_10]
width = 0.3
xs = [0,
      1-0.5*width, 1+0.5*width,
      2-0.5*width, 2+0.5*width]
colors = ['k', 'g', 'b', 'g', 'b']

thresh_means, thresh_stds = [], []
for ag in comparison_exps:
    m, s = get_threshold(ag)
    thresh_means.append(m)
    thresh_stds.append(s)

plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=15)

f, ax = plt.subplots(1, 1, figsize=(8, 3))
bars = ax.bar(xs, thresh_means, width=width, yerr=thresh_stds, color=colors,
              error_kw=dict(ecolor='m', lw=1.5, capsize=4, capthick=1.5))

ax.set_xticks(np.arange(3))
ax.set_xticklabels([r'$N=1$', r'$N=5$', r'$N=10$'])
ax.set_ylabel('Steps until solved')
yfmt = ticker.ScalarFormatter()
yfmt.set_powerlimits((0, 0))
ax.yaxis.set_major_formatter(yfmt)

for l in ax.get_xticklabels():
    l.set_multialignment('center')

handles = [
    mpatches.Patch(color='k', label='Double Q-learning'),
    mpatches.Patch(color='g', label='Multistep Double Q-learning'),
    mpatches.Patch(color='b', label='MAQL (ours)'),
]

# ax.legend(handles=handles, loc='upper center', bbox_to_anchor=(1.3, 0.8), ncol=1)
ax.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, 1.45), ncol=2)

f.savefig(os.path.join(SAVE_FOLDER, 'imswingup1_comparison.png'), bbox_inches='tight', dpi=200)
plt.close(f)
