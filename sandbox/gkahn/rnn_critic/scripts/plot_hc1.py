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

dqn_1 = load_experiments([53, 54, 55], {'color': 'k', 'label': r'$Q$', 'linestyle': '-'})

dqn_5_et_mean = load_experiments([70, 71, 72], {'color': 'g', 'label': r'mean', 'linestyle': '-'})
dqn_5_et_final = load_experiments([106, 107, 108], {'color': 'g', 'label': r'final', 'linestyle': '-.'})
dqn_5_et_exp_25 = load_experiments([109, 110, 111], {'color': 'g', 'label': r'exponential', 'linestyle': ':'})
dqn_5_et_exp_50 = load_experiments([112, 113, 114], {'color': 'g', 'label': r'exponential', 'linestyle': ':'})
dqn_5_et_exp_75 = load_experiments([115, 116, 117], {'color': 'g', 'label': r'exponential', 'linestyle': ':'})
dqn_5_et_exp_95 = load_experiments([118, 119, 120], {'color': 'g', 'label': r'exponential', 'linestyle': ':'})
dqn_5_et_exps = [dqn_5_et_exp_25, dqn_5_et_exp_50, dqn_5_et_exp_75, dqn_5_et_exp_95]
dqn_5_et_exp = dqn_5_et_exps[np.argmax([np.mean([exp[1][-1] for exp in exps]) for exps in dqn_5_et_exps])]
mac_5_et_mean = load_experiments([74, 75, 76], {'color': 'b', 'label': r'MAC', 'linestyle': '-'})
et_5_exps = [dqn_1, dqn_5_et_mean, dqn_5_et_final, dqn_5_et_exp, mac_5_et_mean]

dqn_5_wb_mean = load_experiments([56, 57, 58], {'color': 'g', 'label': r'mean', 'linestyle': '-'})
dqn_5_wb_final = load_experiments([91, 92, 93], {'color': 'g', 'label': r'final', 'linestyle': '-.'})
dqn_5_wb_exp_25 = load_experiments([94, 95, 96], {'color': 'g', 'label': r'exponential', 'linestyle': ':'})
dqn_5_wb_exp_50 = load_experiments([97, 98, 99], {'color': 'g', 'label': r'exponential', 'linestyle': ':'})
dqn_5_wb_exp_75 = load_experiments([100, 101, 102], {'color': 'g', 'label': r'exponential', 'linestyle': ':'})
dqn_5_wb_exp_95 = load_experiments([103, 104, 105], {'color': 'g', 'label': r'exponential', 'linestyle': ':'})
dqn_5_wb_exps = [dqn_5_wb_exp_25, dqn_5_wb_exp_50, dqn_5_wb_exp_75, dqn_5_wb_exp_95]
dqn_5_wb_exp = dqn_5_wb_exps[np.argmax([np.mean([exp[1][-1] for exp in exps]) for exps in dqn_5_wb_exps])]
mac_5_wb_mean = load_experiments([48, 49], {'color': 'b', 'label': r'MAC', 'linestyle': '-'})
wb_5_exps = [dqn_1, dqn_5_wb_mean, dqn_5_wb_final, dqn_5_wb_exp, mac_5_wb_mean]

dqn_10_et_mean = load_experiments([77, 78, 79], {'color': 'g', 'label': r'mean', 'linestyle': '-'})
dqn_10_et_final = load_experiments([136, 137, 138], {'color': 'g', 'label': r'final', 'linestyle': '-.'})
dqn_10_et_exp_25 = load_experiments([139, 140, 141], {'color': 'g', 'label': r'exponential', 'linestyle': ':'})
dqn_10_et_exp_50 = load_experiments([142, 143, 144], {'color': 'g', 'label': r'exponential', 'linestyle': ':'})
dqn_10_et_exp_75 = load_experiments([145, 146, 147], {'color': 'g', 'label': r'exponential', 'linestyle': ':'})
dqn_10_et_exp_95 = load_experiments([148, 149, 150], {'color': 'g', 'label': r'exponential', 'linestyle': ':'})
dqn_10_et_exps = [dqn_10_et_exp_25, dqn_10_et_exp_50, dqn_10_et_exp_75, dqn_10_et_exp_95]
dqn_10_et_exp = dqn_10_et_exps[np.argmax([np.mean([exp[1][-1] for exp in exps]) for exps in dqn_10_et_exps])]
mac_10_et_mean = load_experiments([80, 81, 82], {'color': 'b', 'label': r'MAC', 'linestyle': '-'})
et_10_exps = [dqn_1, dqn_10_et_mean, dqn_10_et_final, dqn_10_et_exp, mac_10_et_mean]

dqn_10_wb_mean = load_experiments([59, 60, 61], {'color': 'g', 'label': r'mean', 'linestyle': '-'})
dqn_10_wb_final = load_experiments([121, 122, 123], {'color': 'g', 'label': r'final', 'linestyle': '-.'})
dqn_10_wb_exp_25 = load_experiments([124, 125, 126], {'color': 'g', 'label': r'exponential', 'linestyle': ':'})
dqn_10_wb_exp_50 = load_experiments([127, 128, 129], {'color': 'g', 'label': r'exponential', 'linestyle': ':'})
dqn_10_wb_exp_75 = load_experiments([130, 131, 132], {'color': 'g', 'label': r'exponential', 'linestyle': ':'})
dqn_10_wb_exp_95 = load_experiments([133, 134, 135], {'color': 'g', 'label': r'exponential', 'linestyle': ':'})
dqn_10_wb_exps = [dqn_10_wb_exp_25, dqn_10_wb_exp_50, dqn_10_wb_exp_75, dqn_10_wb_exp_95]
dqn_10_wb_exp = dqn_10_wb_exps[np.argmax([np.mean([exp[1][-1] for exp in exps]) for exps in dqn_10_wb_exps])]
mac_10_wb_mean = load_experiments([50, 51], {'color': 'b', 'label': r'MAC', 'linestyle': '-'})
wb_10_exps = [dqn_1, dqn_10_wb_mean, dqn_10_wb_final, dqn_10_wb_exp, mac_10_wb_mean]

# 4 columns [N=5,ET] [N=5,WB], [N=10,ET], [N=10,WB]

############
### Plot ###
############

def plot_cumreward(ax, analyze_group, window=50):
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

    max_step = min(1.5e6, max_step)

    steps = np.r_[min_step:max_step:50.][1:-1]
    values_mean, values_std = data_interp.eval(steps)
    # steps -= min_step

    ax.plot(steps, values_mean, **plot)
    # ax.fill_between(steps, values_mean - values_std, values_mean + values_std,
    #                 color=color, alpha=0.4)

    ax.grid()
    xfmt = ticker.ScalarFormatter()
    xfmt.set_powerlimits((0, 0))
    ax.xaxis.set_major_formatter(xfmt)

comparison_exps = [et_5_exps, wb_5_exps, et_10_exps, wb_10_exps]
titles = [r'ET', r'CMR (ours)',
          r'ET', r'CMR (ours)']

import IPython; IPython.embed()

plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=18)

f = plt.figure(figsize=(12, 2))
gs1 = GridSpec(1, 2)
gs1.update(left=0.05, right=0.45, wspace=0.05)
ax1 = plt.subplot(gs1[0])
ax2 = plt.subplot(gs1[1], sharex=ax1, sharey=ax1)

gs2 = GridSpec(1, 2)
gs2.update(left=0.55, right=0.95, wspace=0.05)
ax3 = plt.subplot(gs2[0], sharex=ax1, sharey=ax1)
ax4 = plt.subplot(gs2[1], sharex=ax1, sharey=ax1)

axes = np.array([ax1, ax2, ax3, ax4])

for i, (ax, exps) in enumerate(zip(axes.ravel(), comparison_exps)):
    for j, exp in enumerate(exps):
        print(i, j)
        plot_cumreward(ax, exp)

handles = [
    mpatches.Patch(color='k', label=r'Standard critic'),
    mlines.Line2D(range(1), range(1), color='k', alpha=0.4, label=r'mean', linestyle='-'),
    mpatches.Patch(color='g', label=r'MRC'),
    mlines.Line2D(range(1), range(1), color='k', alpha=0.4, label=r'final', linestyle='-.'),
    mpatches.Patch(color='b', label=r'MAC (ours)'),
    mlines.Line2D(range(1), range(1), color='k', alpha=0.4, label=r'exponential', linestyle='--'),
]

axes[1].legend(handles=handles, ncol=3, loc='upper right', bbox_to_anchor=(2.7, 2.0))

for ax, title in zip(axes.ravel(), titles):
    ax.set_title(title)
    ax.set_xlabel('Steps')
plt.setp(axes[1].get_yticklabels(), visible=False)
plt.setp(axes[3].get_yticklabels(), visible=False)
axes[0].set_ylabel(r'Cumulative Reward')
axes[2].set_ylabel(r'Cumulative Reward')

artists = [axes[0].text(1., -0.7, r'(i) $N=5$', ha='center', transform=axes[0].transAxes),
           axes[2].text(1., -0.7, r'(ii) $N=10$', ha='center', transform=axes[2].transAxes)]

f.savefig(os.path.join(SAVE_FOLDER, 'hc1_comparison.png'), bbox_inches='tight', dpi=300, bbox_artists=artists)
plt.close(f)


