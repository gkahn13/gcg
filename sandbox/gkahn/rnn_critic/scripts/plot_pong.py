#type_replacements = [
#    {'V_N': 1, 'V_H': 1, 'V_action': 'lattice', 'V_class': 'DQNPolicy', 'V_num_offpolicy': '1.e+4', 'G0': 3, 'G1': 0.3}, # DQN
#    {'V_N': 1, 'V_H': 1, 'V_action': 'lattice', 'V_class': 'DQNPolicy', 'V_num_offpolicy': '1.e+5', 'G0': 3, 'G1': 0.3}, # DQN
#    {'V_N': 1, 'V_H': 1, 'V_action': 'lattice', 'V_class': 'DQNPolicy', 'V_num_offpolicy': '1.e+6', 'G0': 3, 'G1': 0.3}, # DQN
#    {'V_N': 5, 'V_H': 5, 'V_action': 'lattice', 'V_class': 'MACMuxPolicy', 'V_num_offpolicy': '1.e+4', 'G0': 4, 'G1': 0.35}, # MAC 5
#    {'V_N': 5, 'V_H': 5, 'V_action': 'lattice', 'V_class': 'MACMuxPolicy', 'V_num_offpolicy': '1.e+5', 'G0': 6, 'G1': 0.35}, # MAC 5
#    {'V_N': 5, 'V_H': 5, 'V_action': 'lattice', 'V_class': 'MACMuxPolicy', 'V_num_offpolicy': '1.e+6', 'G0': 7, 'G1': 0.35}, # MAC 5
#    {'V_N': 10, 'V_H': 10, 'V_action': 'random', 'V_class': 'MACMuxPolicy', 'V_num_offpolicy': '1.e+4', 'G0': 4, 'G1': 0.55}, # MAC 10
#    {'V_N': 10, 'V_H': 10, 'V_action': 'random', 'V_class': 'MACMuxPolicy', 'V_num_offpolicy': '1.e+5', 'G0': 6, 'G1': 0.55}, # MAC 10
#    {'V_N': 10, 'V_H': 10, 'V_action': 'random', 'V_class': 'MACMuxPolicy', 'V_num_offpolicy': '1.e+6', 'G0': 7, 'G1': 0.55}, # MAC 10
#]

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
        print(i)
        exps.append(AnalyzeRNNCritic(os.path.join(EXP_FOLDER, 'pong{0:03d}'.format(i)),
                                     clear_obs=False,
                                     create_new_envs=False))
    return exps

dqn_1_1e5 = load_experiments([112, 113, 114])
dqn_5_1e5 = load_experiments([131, 132, 133])
dqn_10_1e5 = load_experiments([134, 135, 136])
mac_5_1e5 = load_experiments([115, 116, 117])
mac_10_1e5 = load_experiments([118, 119, 120])

dqn_1_1e6 = load_experiments([121, 123]) # 122
dqn_5_1e6 = load_experiments([137, 138, 139])
dqn_10_1e6 = load_experiments([140, 141, 142])
mac_5_1e6 = load_experiments([124, 125, 126])
mac_10_1e6 = load_experiments([127, 128, 129])


dqn_5_1e5_one_mse = load_experiments([151])
mac_5_1e5_one_mse = load_experiments([153])
dqn_10_1e5_one_mse = load_experiments([155])
mac_10_1e5_one_mse = load_experiments([157])

dqn_5_1e6_one_mse = load_experiments([152])
mac_5_1e6_one_mse = load_experiments([154])
dqn_10_1e6_one_mse = load_experiments([156])
mac_10_1e6_one_mse = load_experiments([158])

comparison_exps = np.array([[dqn_1_1e5, dqn_5_1e5, dqn_5_1e5_one_mse, dqn_10_1e5, dqn_10_1e5_one_mse, mac_5_1e5, mac_5_1e5_one_mse, mac_10_1e5, mac_10_1e5_one_mse],
                            [dqn_1_1e6, dqn_5_1e6, dqn_5_1e6_one_mse, dqn_10_1e6, dqn_10_1e6_one_mse, mac_5_1e6, mac_5_1e6_one_mse, mac_10_1e6, mac_10_1e6_one_mse]])

############
### Plot ###
############

def plot_cumreward(ax, analyze_group, color='k', label=None, window=100):
    data_interp = DataAverageInterpolation()
    min_step = max_step = None
    for analyze in analyze_group:
        rollouts = list(itertools.chain(*analyze.eval_rollouts_itrs))
        rollouts = sorted(rollouts, key=lambda r: r['steps'][0])
        steps = [r['steps'][0] for r in rollouts]
        values = [np.sum(r['rewards']) for r in rollouts]

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

    ax.plot(steps, values_mean, color=color, label=label)
    ax.fill_between(steps, values_mean - values_std, values_mean + values_std,
                    color=color, alpha=0.4)

    ax.grid()
    xfmt = ticker.ScalarFormatter()
    xfmt.set_powerlimits((0, 0))
    ax.xaxis.set_major_formatter(xfmt)

shape = comparison_exps.shape[:2]
f, axes = plt.subplots(*shape, figsize=(15, 5), sharex=True, sharey=True)
for i in range(shape[0]):
    for j in range(shape[1]):
        plot_cumreward(axes[i, j], comparison_exps[i, j], window=50)

for i, N in enumerate(['1e5', '1e6']):
    axes[i, 0].set_ylabel('num_offpolicy = {0}'.format(N))
for j, name in enumerate(['DQN', 'DQN-5', 'DQN-5 one mse', 'DQN-10', 'DQN-10 one mse', 'MAC-5', 'MAC-5 one mse', 'MAC-10', 'MAC-10 one mse']):
    axes[0, j].set_title(name)

# plt.tight_layout()
f.savefig(os.path.join(SAVE_FOLDER, 'pong_offpolicy_comparison.png'), bbox_inches='tight', dpi=200)
plt.close(f)

import IPython; IPython.embed()
