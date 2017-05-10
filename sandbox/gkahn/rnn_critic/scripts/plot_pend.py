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

def load_experiments(start, repeat=3):
    exps = []
    for i in range(start, start + repeat):
        exps.append(AnalyzeRNNCritic(os.path.join(EXP_FOLDER, 'pend{0:03d}'.format(i)),
                                     clear_obs=False,
                                     create_new_envs=False))
    return exps

dqn_1 = load_experiments(157)
dqn_5 = load_experiments(148)
dqn_10 = load_experiments(151)
dqn_20 = load_experiments(161)

predictron_1 = load_experiments(86)
predictron_5 = load_experiments(141)
predictron_10 = load_experiments(144)
predictron_20 = load_experiments(164)

mac_1 = load_experiments(86)
mac_5 = load_experiments(110)
mac_10 = load_experiments(134)
mac_20 = load_experiments(167)

mac_10_test_1_targ_1 = load_experiments(116)
mac_10_test_10_targ_1 = load_experiments(122)
mac_10_test_1_targ_10 = load_experiments(128)
mac_10_test_10_targ_10 = load_experiments(134)

comparison_exps = np.array([[dqn_1, dqn_5, dqn_10, dqn_20],
                            [predictron_1, predictron_5, predictron_10, predictron_20],
                            [mac_1, mac_5, mac_10, mac_20]])

ablation_exps = np.array([[mac_10_test_1_targ_1, mac_10_test_10_targ_1, mac_10_test_1_targ_10, mac_10_test_10_targ_10]])

############
### Plot ###
############

def plot_cumreward(ax, analyze_group, color='k', label=None):
    data_interp = DataAverageInterpolation()
    min_step = max_step = None
    for analyze in analyze_group:
        rollouts = list(itertools.chain(*analyze.eval_rollouts_itrs))
        rollouts = sorted(rollouts, key=lambda r: r['steps'][-1])
        steps = [r['steps'][-1] for r in rollouts]
        values = [np.sum(r['rewards']) for r in rollouts]

        def moving_avg_std(idxs, data, window):
            avg_idxs, means, stds = [], [], []
            for i in range(window, len(data)):
                avg_idxs.append(np.mean(idxs[i - window:i]))
                means.append(np.mean(data[i - window:i]))
                stds.append(np.std(data[i - window:i]))
            return avg_idxs, np.asarray(means), np.asarray(stds)

        steps, values, _ = moving_avg_std(steps, values, window=10)

        data_interp.add_data(steps, values)
        if min_step is None:
            min_step = steps[0]
        if max_step is None:
            max_step = steps[-1]
        min_step = max(min_step, steps[0])
        max_step = min(max_step, steps[-1])

    min_step = analyze.params['alg']['learn_after_n_steps']

    steps = np.r_[min_step:max_step:50.][1:-1]
    values_mean, values_std = data_interp.eval(steps)
    steps -= min_step

    ax.plot(steps, values_mean, color=color, label=label)
    ax.fill_between(steps, values_mean - values_std, values_mean + values_std,
                    color=color, alpha=0.4)

    threshold = -250
    if values_mean.max() > threshold:
        ax.vlines(steps[(values_mean > threshold).argmax()], *ax.get_ylim(), color='g', linestyle='--')

    ax.grid()
    xfmt = ticker.ScalarFormatter()
    xfmt.set_powerlimits((0, 0))
    ax.xaxis.set_major_formatter(xfmt)

shape = comparison_exps.shape[:2]
f, axes = plt.subplots(*shape, figsize=(15, 5), sharex=True, sharey=True)
for i in range(shape[0]):
    for j in range(shape[1]):
        plot_cumreward(axes[i, j], comparison_exps[i, j, :])
        axes[i, j].set_ylim((-1600, 0))

for j, N in enumerate([1, 5, 10, 20]):
    axes[0, j].set_title('N = {0}'.format(N))
for i, name in enumerate(['DQN', 'Predictron', 'MAC']):
    axes[i, 0].set_ylabel(name)

# plt.tight_layout()
f.savefig(os.path.join(SAVE_FOLDER, 'pend_comparison.png'), bbox_inches='tight', dpi=200)
plt.close(f)

import IPython; IPython.embed()
