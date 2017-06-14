
### N = 20
### For each method, best ET and best WB (line style of bar)
### horizontal bar graph, each color is a method

import os
import numpy as np
import itertools
from collections import OrderedDict as OD

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import ticker
import matplotlib.patches as mpatches

from analyze_experiment import AnalyzeRNNCritic
from sandbox.gkahn.rnn_critic.utils.utils import DataAverageInterpolation

EXP_FOLDER = '/media/gkahn/ExtraDrive1/rllab/s3/rnn-critic/'
SAVE_FOLDER = '/media/gkahn/ExtraDrive1/rllab/rnn_critic/final_plots'

########################
### Load experiments ###
########################

CUM_REWARD_THRESHOLD = -250

def process_experiments(start_index, repeat, window=10):
    analyze_names = []
    data_interp = DataAverageInterpolation()
    min_step = max_step = None
    above_threshold_steps = []
    for index in range(start_index, start_index + repeat):
        try:
            analyze = AnalyzeRNNCritic(os.path.join(EXP_FOLDER, 'pend{0:03d}'.format(index)), clear_obs=False, create_new_envs=False)
        except:
            continue

        # pend612	V_class: MACPolicy, V_N: 20, V_H: 20, V_test_H: 20, V_target_H: 20, V_softmax: exponential, V_exp_lambda: 0.75, V_retrace_lambda: , V_use_target: True, V_share_weights: True,
        analyze_name = '{0: <25}, N: {1: <2}, H: {2: <2}, use_target: {3: <2}'.format(analyze.params['policy']['class'],
                                                                                      analyze.params['policy']['N'],
                                                                                      analyze.params['policy']['H'],
                                                                                      analyze.params['policy']['use_target'])
        if analyze.params['policy']['values_softmax']['type'] == 'exponential':
            analyze_name += ', values_softmax: {0: <20}'.format(analyze.params['policy']['values_softmax']['type'] + '(' + \
                                                                str(analyze.params['policy']['values_softmax']['exponential']['lambda']) + ')')
        else:
            analyze_name += ', values_softmax: {0: <20}'.format(analyze.params['policy']['values_softmax']['type'])
        rt = analyze.params['policy']['retrace_lambda']
        analyze_name += ', retrace: {0: <5}'.format(rt if rt else '')
        if analyze.params['policy']['class'] == 'MACPolicy':
            analyze_name += ', share_weights: {0: <5}'.format(analyze.params['policy']['MACPolicy']['share_weights'])
        else:
            analyze_name += ', share_weights: {0: <5}'.format(' ')
        analyze_name += ', separate_mses: {0: <3}'.format(analyze.params['policy'].get('separate_mses', True))
        if len(analyze_names) > 0:
            assert(analyze_name == analyze_names[-1])
        analyze_names.append(analyze_name)

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

        if len(analyze_names) == 0:
            raise Exception

        steps, values, _ = moving_avg_std(steps, values, window=window)

        if values.max() > CUM_REWARD_THRESHOLD:
            above_threshold_steps.append(steps[(values > CUM_REWARD_THRESHOLD).argmax()])
        else:
            above_threshold_steps.append(steps[-1])

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

    if values_mean.max() > CUM_REWARD_THRESHOLD:
        above_threshold_step = steps[(values_mean > CUM_REWARD_THRESHOLD).argmax()]
    else:
        above_threshold_step = steps[-1]

    # return analyze_name, above_threshold_step, 0
    return analyze_name, np.mean(above_threshold_steps), np.std(above_threshold_steps)

names, thresholds = [], []
edict = {}
for i in [201, 694, 627, 573, 712, 651, 579, 730, 675, 612, 321, 327, 348, 489, 495, 531]:
    try:
        name, threshold, threshold_std = process_experiments(i, 3, window=10)
        names.append('pend{0:03g} '.format(i) + name)
        thresholds.append(threshold)
        edict[i] = (threshold, threshold_std)
    except:
        print('Failed to load {0}'.format(i))
        edict[i] = np.nan

import IPython; IPython.embed()

############
### Plot ###
############

Q_thresh = edict[201]

Q_5_N_thresh = edict[694]
Q_5_rnn_thresh = edict[627]
Q_5_ff_mac_thresh = edict[321]
Q_5_nws_mac_thresh = edict[489]
Q_5_mac_thresh = edict[573]

Q_10_N_thresh = edict[712]
Q_10_rnn_thresh = edict[651]
Q_10_ff_mac_thresh = edict[327]
Q_10_nws_mac_thresh = edict[495]
Q_10_mac_thresh = edict[579]

Q_20_N_thresh = edict[730]
Q_20_rnn_thresh = edict[675]
Q_20_ff_mac_thresh = edict[348]
Q_20_nws_mac_thresh = edict[531]
Q_20_mac_thresh = edict[612]

width = 0.15
xs = [0,
      1-2.*width, 1-1.*width, 1, 1+1.*width, 1+2.*width,
      2-2.*width, 2-1.*width, 2, 2+1.*width, 2+2.*width,
      3-2.*width, 3-1.*width, 3, 3+1.*width, 3+2.*width]
thresh_means, thresh_stds = zip(*(Q_thresh,
                                  Q_5_N_thresh, Q_5_rnn_thresh, Q_5_ff_mac_thresh, Q_5_nws_mac_thresh, Q_5_mac_thresh,
                                  Q_10_N_thresh, Q_10_rnn_thresh, Q_10_ff_mac_thresh, Q_10_nws_mac_thresh, Q_10_mac_thresh,
                                  Q_20_N_thresh, Q_20_rnn_thresh, Q_20_ff_mac_thresh, Q_20_nws_mac_thresh, Q_20_mac_thresh))
colors = ['k'] + ['g', 'c', 'b', 'b', 'b'] * 3

plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=15)
mpl.rcParams['hatch.color'] = (1, 1, 1)

f, ax = plt.subplots(1, 1, figsize=(10, 2))
bars = ax.bar(xs, thresh_means, width=width, yerr=thresh_stds, color=colors,
              error_kw=dict(ecolor='m', lw=1.5, capsize=4, capthick=1.5))

ax.set_xticks(np.arange(4))
ax.set_xticklabels([r'$N=1$', r'$N=5$', r'$N=10$', r'$N=20$'])
ax.set_ylabel('Steps until solved')
yfmt = ticker.ScalarFormatter()
yfmt.set_powerlimits((0, 0))
ax.yaxis.set_major_formatter(yfmt)

for i in (1, 6, 11):
    bars[i + 2].set_hatch('//')
    bars[i + 3].set_hatch('o')

handles = [
    mpatches.Patch(color='k', label='Double Q-learning'),
    mpatches.Patch(color='g', label='Multistep Double Q-learning'),
    mpatches.Patch(color='c', label='Multistep RNN Double Q-learning'),
    mpatches.Patch(color='b', label='MAQL feedforward', hatch='//', edgecolor='white'),
    mpatches.Patch(color='b', label='MAQL no weight sharing', hatch='oo', edgecolor='white'),
    mpatches.Patch(color='b', label='MAQL (ours)'),
]

# ax.legend(handles=handles, loc='upper center', bbox_to_anchor=(1.25, 0.8), ncol=1)
ax.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.52, 1.75), ncol=2)

f.savefig(os.path.join(SAVE_FOLDER, 'pend3_comparison.png'), bbox_inches='tight', dpi=200)
plt.close(f)
