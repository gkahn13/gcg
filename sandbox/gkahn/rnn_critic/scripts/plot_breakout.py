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
            analyze = AnalyzeRNNCritic(os.path.join(EXP_FOLDER, 'breakout{0:03d}'.format(i)),
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

#breakout001	V_class: DQNPolicy, V_separate_mses: True, V_N: 1, V_H: 1, V_action_type: lattice, G0: 0, G1: 0.25,
#breakout002	V_class: DQNPolicy, V_separate_mses: True, V_N: 1, V_H: 1, V_action_type: lattice, G0: 0, G1: 0.25,
#breakout003	V_class: DQNPolicy, V_separate_mses: True, V_N: 1, V_H: 1, V_action_type: lattice, G0: 0, G1: 0.25,
#breakout004	V_class: DQNPolicy, V_separate_mses: True, V_N: 5, V_H: 1, V_action_type: lattice, G0: 4, G1: 0.3,
#breakout005	V_class: DQNPolicy, V_separate_mses: True, V_N: 5, V_H: 1, V_action_type: lattice, G0: 4, G1: 0.3,
#breakout006	V_class: DQNPolicy, V_separate_mses: True, V_N: 5, V_H: 1, V_action_type: lattice, G0: 4, G1: 0.3,
#breakout007	V_class: DQNPolicy, V_separate_mses: False, V_N: 5, V_H: 1, V_action_type: lattice, G0: 5, G1: 0.3,
#breakout008	V_class: DQNPolicy, V_separate_mses: False, V_N: 5, V_H: 1, V_action_type: lattice, G0: 5, G1: 0.3,
#breakout009	V_class: DQNPolicy, V_separate_mses: False, V_N: 5, V_H: 1, V_action_type: lattice, G0: 5, G1: 0.3,
#breakout010	V_class: DQNPolicy, V_separate_mses: True, V_N: 10, V_H: 1, V_action_type: lattice, G0: 6, G1: 0.3,
#breakout011	V_class: DQNPolicy, V_separate_mses: True, V_N: 10, V_H: 1, V_action_type: lattice, G0: 6, G1: 0.3,
#breakout012	V_class: DQNPolicy, V_separate_mses: True, V_N: 10, V_H: 1, V_action_type: lattice, G0: 6, G1: 0.3,
#breakout013	V_class: DQNPolicy, V_separate_mses: False, V_N: 10, V_H: 1, V_action_type: lattice, G0: 7, G1: 0.3,
#breakout014	V_class: DQNPolicy, V_separate_mses: False, V_N: 10, V_H: 1, V_action_type: lattice, G0: 7, G1: 0.3,
#breakout015	V_class: DQNPolicy, V_separate_mses: False, V_N: 10, V_H: 1, V_action_type: lattice, G0: 7, G1: 0.3,

dqn_1 = load_experiments([1, 2, 3])
dqn_5_wb = load_experiments([4, 5, 6])
dqn_5_et = load_experiments([7, 8, 9])
dqn_10_wb = load_experiments([10, 11, 12])
dqn_10_et = load_experiments([13, 14, 15])

import IPython; IPython.embed()

############
### Plot ###
############

def plot_cumreward(ax, analyze_group, color='k', label=None, window=10  ):
    data_interp = DataAverageInterpolation()
    min_step = max_step = None
    for i, (steps, values) in enumerate(analyze_group):
        steps = 4 * np.array(steps) # frame skip

        def moving_avg_std(idxs, data, window):
            avg_idxs, means, stds = [], [], []
            for i in range(window, len(data)):
                avg_idxs.append(np.mean(idxs[i - window:i]))
                means.append(np.mean(data[i - window:i]))
                stds.append(np.std(data[i - window:i]))
            return avg_idxs, np.asarray(means), np.asarray(stds)

        steps, values, _ = moving_avg_std(steps, values, window=window)

        # ax.plot(steps, values, color='k', alpha=np.linspace(1., 0.4, len(analyze_group))[i])

        if len(steps) == 0:
            continue

        data_interp.add_data(steps, values)
        if min_step is None:
            min_step = steps[0]
        if max_step is None:
            max_step = steps[-1]
        min_step = max(min_step, steps[0])
        max_step = min(max_step, steps[-1])

    if len(analyze_group) > 1:
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
f.savefig(os.path.join(SAVE_FOLDER, 'breakout_comparison.png'), bbox_inches='tight', dpi=300)
plt.close(f)

