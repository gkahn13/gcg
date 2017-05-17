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
dqn_5_et = load_experiments([21, 22, 23])
dqn_5_wb = load_experiments([6, 7, 8])
dqn_10_et = load_experiments([27, 28, 29])
dqn_10_wb = load_experiments([16, 17, 18])
# rnn_5_wb = load_experiments([9, 10, 11])
mac_5_et = load_experiments([24, 25, 26])
mac_5_wb = load_experiments([1, 2, 3])
mac_10_wb = load_experiments([12, 13, 14])

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

comparison_exps = [dqn_1, dqn_5_et, dqn_5_wb, mac_5_et, mac_5_wb]
width = 0.4
xs = [0,
      1-1.05*width/2., 1+1.05*width/2.,
      2-1.05*width/2., 2+1.05*width/2.]
colors = ['k', 'g', 'g', 'b', 'b']
patterns = [None, '/', '+', '/', '+']

thresh_means, thresh_stds = [], []
for ag in comparison_exps:
    m, s = get_threshold(ag)
    thresh_means.append(m)
    thresh_stds.append(s)

plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=15)

f, ax = plt.subplots(1, 1, figsize=(7, 3))
bars = ax.bar(xs, thresh_means, width=width, yerr=thresh_stds, color=colors,
              error_kw=dict(ecolor='gray', lw=1, capsize=2, capthick=1))

ax.set_xticks(np.arange(3))
ax.set_xticklabels([r'Standard Critic', r'MRC', r'MAC (ours)', ''])
ax.set_ylabel('Steps until solved')
yfmt = ticker.ScalarFormatter()
yfmt.set_powerlimits((0, 0))
ax.yaxis.set_major_formatter(yfmt)

for bar, pattern in zip(bars, patterns):
    bar.set_hatch(pattern)

et_patch = mpatches.Patch(color='k', alpha=0.4, label='ET', hatch='///')
wb_patch = mpatches.Patch(color='k', alpha=0.4, label='CMR (ours)', hatch='+++')
ax.legend(handles=(et_patch, wb_patch))

for l in ax.get_xticklabels():
    l.set_multialignment('center')

f.savefig(os.path.join(SAVE_FOLDER, 'imswingup1_comparison.png'), bbox_inches='tight', dpi=200)
plt.close(f)
