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

def load_experiments(indices):
    exps = []
    for i in indices:
        try:
            exps.append(AnalyzeRNNCritic(os.path.join(EXP_FOLDER, 'phd{0:03d}'.format(i)),
                                         clear_obs=False,
                                         create_new_envs=False))
        except:
            pass
    return exps

dqn_1 = load_experiments([274, 275, 276])
dqn_4 = load_experiments([277, 278, 279])
dqn_8 = load_experiments([280, 281, 282])

dqn_retrace_1 = load_experiments([283, 284, 285])
dqn_retrace_4 = load_experiments([286, 287, 288])
dqn_retrace_8 = load_experiments([289, 290, 291])

predictron_1 = load_experiments([292, 293, 294])
predictron_4 = load_experiments([295, 296, 297])
predictron_8 = load_experiments([298, 299, 300])

mb_1 = load_experiments([301, 302, 303])
mb_4 = load_experiments([304, 305, 306])
mb_8 = load_experiments([307, 308, 309])

mac_1 = load_experiments([292, 293, 294])
mac_4 = load_experiments([310, 311, 312])
mac_8 = load_experiments([313, 314, 315])

import IPython; IPython.embed()

comparison_exps = np.array([[dqn_1, dqn_retrace_1, predictron_1, mb_1, mac_1],
                            [dqn_4, dqn_retrace_4, predictron_4, mb_4, mac_4],
                            [dqn_8, dqn_retrace_8, predictron_8, mb_4, mac_8]])

############
### Plot ###
############

def plot_episodelength(ax, analyze_group, window=100):
    data_interp = DataAverageInterpolation()
    min_step = max_step = None
    for analyze in analyze_group:
        rollouts = list(itertools.chain(*analyze.eval_rollouts_itrs))
        rollouts = sorted(rollouts, key=lambda r: r['steps'][0])
        steps = [r['steps'][0] for r in rollouts]
        values = [len(r['rewards']) for r in rollouts]

        def moving_avg_std(idxs, data, window):
            avg_idxs, means, stds = [], [], []
            for i in range(window, len(data)):
                avg_idxs.append(np.mean(idxs[i - window:i]))
                means.append(np.mean(data[i - window:i]))
                stds.append(np.std(data[i - window:i]))
            return avg_idxs, np.asarray(means), np.asarray(stds)

        steps, values, _ = moving_avg_std(steps, values, window=window)

        data_interp.add_data(steps, values)
        if min_step is None:
            min_step = steps[0]
        if max_step is None:
            max_step = steps[-1]
        min_step = max(min_step, steps[0])
        max_step = min(max_step, steps[-1])

    steps = np.r_[min_step:max_step:50.][1:-1]
    values_mean, values_std = data_interp.eval(steps)
    steps -= min_step

    ax.plot(steps, values_mean, color='k')
    ax.fill_between(steps, values_mean - values_std, values_mean + values_std,
                    color='k', alpha=0.4)

shape = comparison_exps.shape[:2]
f, axes = plt.subplots(*shape, figsize=(15, 5), sharex=True, sharey=True)
for i in range(shape[0]):
    for j in range(shape[1]):
        if len(comparison_exps[i, j]) > 0:
            try:
                plot_episodelength(axes[i, j], comparison_exps[i, j], window=50)
            except:
                pass

for i, N in enumerate([1, 4, 8]):
    axes[i, 0].set_ylabel('N = {0}'.format(N))
for j, name in enumerate(['DQN', 'DQN-retrace', 'Predictron', 'Model-based', 'MAC']):
    axes[0, j].set_title(name)

# plt.tight_layout()
f.savefig(os.path.join(SAVE_FOLDER, 'phd_comparison.png'), bbox_inches='tight', dpi=200)
plt.close(f)
