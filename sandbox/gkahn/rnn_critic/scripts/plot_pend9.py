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

def load_experiments(indices, plot=None):
    exps = []
    for i in indices:
        try:
            exps.append(AnalyzeRNNCritic(os.path.join(EXP_FOLDER, 'pend{0:03d}'.format(i)),
                                         clear_obs=False,
                                         create_new_envs=False,
                                         load_train_rollouts=False,
                                         plot=plot))
            print(i)
        except:
            pass

    return exps


cont_dense_dqn = load_experiments(range(941, 944), plot={'color': 'k'})
cont_dense_nstep = load_experiments(range(944, 947), plot={'color': 'g'})
cont_dense_mac = load_experiments(range(947, 950), plot={'color': 'b'})

cont_sparse_dqn = load_experiments(range(956, 959), plot={'color': 'k'})
cont_sparse_nstep = load_experiments(range(959, 962), plot={'color': 'g'})
cont_sparse_mac = load_experiments(range(962, 965), plot={'color': 'b'})

disc_dense_dqn = load_experiments(range(1005, 1008), plot={'color': 'k'})
disc_dense_nstep = load_experiments(range(1008, 1011), plot={'color': 'g'})
disc_dense_mac = load_experiments(range(1011, 1014), plot={'color': 'b'})

disc_sparse_dqn = load_experiments(range(1023, 1026), plot={'color': 'k'})
disc_sparse_nstep = load_experiments(range(1026, 1029), plot={'color': 'g'})
disc_sparse_mac = load_experiments(range(1030, 1033), plot={'color': 'b'})

all_exps = [cont_dense_dqn, cont_dense_nstep, cont_dense_mac,
            cont_sparse_dqn, cont_sparse_nstep, cont_sparse_mac,
            disc_dense_dqn, disc_dense_nstep, disc_dense_mac,
            disc_sparse_dqn, disc_sparse_nstep, disc_sparse_mac]

# import IPython; IPython.embed()

############
### Plot ###
############

CUM_REWARD_THRESHOLD = -250

def plot_cumreward(ax, analyze_group, color='k', label=None, window=20):
    data_interp = DataAverageInterpolation()
    min_step = max_step = None
    for i, analyze in enumerate(analyze_group):
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

    if values_mean.max() > CUM_REWARD_THRESHOLD:
        ax.vlines(steps[(values_mean > CUM_REWARD_THRESHOLD).argmax()], -2000, 0, color='k', alpha=0.5, linestyle='--')

    ax.grid()
    xfmt = ticker.ScalarFormatter()
    xfmt.set_powerlimits((0, 0))
    ax.xaxis.set_major_formatter(xfmt)

plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=15)

f, axes = plt.subplots(3, 4, figsize=(12, 6), sharex=True, sharey=True)

axes = axes.T

for ax, exps in zip(axes.ravel(), all_exps):
    plot_cumreward(ax, exps, color=exps[0].plot['color'])
    # params = exps[0].params
    # ax.set_title('{0}\n{1}, N: {2}, H: {3}'.format(params['alg']['env'].split('(')[1], params['policy']['class'],
    #                                                params['policy']['N'], params['policy']['H']))


axes.T[0, 0].set_title('Continuous actions\nDense rewards')
axes.T[0, 1].set_title('Continuous actions\nSparse rewards')
axes.T[0, 2].set_title('Discrete actions\nDense rewards')
axes.T[0, 3].set_title('Discrete actions\nSparse rewards')

axes.T[0, 1].legend(loc='upper center', ncol=2, bbox_to_anchor=(-0.15, 1.55))

for ax in axes.T[-1, :]:
    ax.set_xlabel('Steps')
for ax in axes.T[:, 0]:
    ax.set_ylabel('Cumulative\nreward')

handles = [
    mpatches.Patch(color='k', label='Double Q-learning', linewidth=0.5),
    mpatches.Patch(color='g', label='Multistep Double Q-learning', linewidth=0.5),
    mpatches.Patch(color='b', label='MAQL (ours)', linewidth=0.5),
]

# ax.legend(handles=handles, loc='upper center', bbox_to_anchor=(1.3, 0.8), ncol=1)
axes.T[0,1].legend(handles=handles, loc='upper center', bbox_to_anchor=(1., 1.78), ncol=3)


f.savefig(os.path.join(SAVE_FOLDER, 'pend9_comparison.png'), bbox_inches='tight', dpi=150)
plt.close(f)

