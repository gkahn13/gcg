import os
import numpy as np
import itertools
from collections import OrderedDict as OD

import matplotlib.pyplot as plt
from matplotlib import ticker

from analyze_experiment import AnalyzeRNNCritic
from sandbox.gkahn.rnn_critic.utils.utils import DataAverageInterpolation

EXP_FOLDER = '/media/gkahn/ExtraDrive1/rllab/s3/rnn-critic/'
SAVE_FOLDER = '/media/gkahn/ExtraDrive1/rllab/rnn_critic/final_plots'

########################
### Load experiments ###
########################

CUM_REWARD_THRESHOLD = -250

def process_experiments_OLD(start_index, repeat, window=10):
    above_threshold_steps = []
    for index in range(start_index, start_index + repeat):
        analyze = AnalyzeRNNCritic(os.path.join(EXP_FOLDER, 'pend{0:03d}'.format(index)), clear_obs=False, create_new_envs=False)

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

        if values.max() > CUM_REWARD_THRESHOLD:
            above_threshold_step = steps[(values > CUM_REWARD_THRESHOLD).argmax()]
        else:
            above_threshold_step = steps[-1]

        above_threshold_step -= analyze.params['alg']['learn_after_n_steps']
        above_threshold_steps.append(above_threshold_step)

    return np.mean(above_threshold_steps), np.std(above_threshold_steps)

def process_experiments(start_index, repeat, window=10):
    analyze_names = []
    data_interp = DataAverageInterpolation()
    min_step = max_step = None
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

    return analyze_name, above_threshold_step, 0

names, thresholds = [], []
edict = {}
for i in list(range(201, 683, 3)) + list(range(685, 846, 3)):
# for i in list(range(685, 846, 3)):
    try:
        name, threshold, threshold_std = process_experiments(i, 3, window=10)
        names.append('pend{0:03g} '.format(i) + name)
        thresholds.append(threshold)
        edict[i] = threshold
    except:
        print('Failed to load {0}'.format(i))
        edict[i] = np.nan

s = ''
for name, threshold in zip(names, thresholds):
    s += '{0:.1f}e3\t{1}\n'.format(threshold/1e3, name)
# print(s)

with open(os.path.join(SAVE_FOLDER, 'plot_pend0.txt'), 'w') as f:
    f.write(s)

def add(d, value_list, final_value):
    curr_d = d
    for i, v in enumerate(value_list[:-1]):
        if v not in curr_d.keys():
            curr_d[v] = OD()
        curr_d = curr_d[v]

    curr_d[value_list[-1]] = final_value

d = OD()
# DQN
add(d, ('Q', '', r"$N=1$", 'mean'),      edict[201])
add(d, ('Q', '', r"$N=1$", 'final'),     edict[201])
add(d, ('Q', '', r"$N=1$", r"$\lambda(0.25)$"), edict[201])
add(d, ('Q', '', r"$N=1$", r"$\lambda(0.5)$"),  edict[201])
add(d, ('Q', '', r"$N=1$", r"$\lambda(0.75)$"), edict[201])
add(d, ('Q', '', r"$N=1$", r"$\lambda(0.9)$"),  edict[201])

# N-step Q (eligibility trace)
add(d, (r"$Q_N$", 'ET', r"$N=5$", 'mean'),      edict[685])
add(d, (r"$Q_N$", 'ET', r"$N=5$", 'final'),     edict[688])
add(d, (r"$Q_N$", 'ET', r"$N=5$", r"$\lambda(0.25)$"), edict[691])
add(d, (r"$Q_N$", 'ET', r"$N=5$", r"$\lambda(0.5)$"),  edict[694])
add(d, (r"$Q_N$", 'ET', r"$N=5$", r"$\lambda(0.75)$"), edict[697])
add(d, (r"$Q_N$", 'ET', r"$N=5$", r"$\lambda(0.9)$"),  edict[700])

add(d, (r"$Q_N$", 'ET', r"$N=10$", 'mean'),      edict[703])
add(d, (r"$Q_N$", 'ET', r"$N=10$", 'final'),     edict[706])
add(d, (r"$Q_N$", 'ET', r"$N=10$", r"$\lambda(0.25)$"), edict[709])
add(d, (r"$Q_N$", 'ET', r"$N=10$", r"$\lambda(0.5)$"),  edict[712])
add(d, (r"$Q_N$", 'ET', r"$N=10$", r"$\lambda(0.75)$"), edict[715])
add(d, (r"$Q_N$", 'ET', r"$N=10$", r"$\lambda(0.9)$"),  edict[718])

add(d, (r"$Q_N$", 'ET', r"$N=20$", 'mean'),      edict[721])
add(d, (r"$Q_N$", 'ET', r"$N=20$", 'final'),     edict[724])
add(d, (r"$Q_N$", 'ET', r"$N=20$", r"$\lambda(0.25)$"), edict[727])
add(d, (r"$Q_N$", 'ET', r"$N=20$", r"$\lambda(0.5)$"),  edict[730])
add(d, (r"$Q_N$", 'ET', r"$N=20$", r"$\lambda(0.75)$"), edict[733])
add(d, (r"$Q_N$", 'ET', r"$N=20$", r"$\lambda(0.9)$"),  edict[736])

# N-step Q (weighted bellman)
add(d, (r"$Q_N$", 'WB', r"$N=5$", 'mean'),      edict[222])
add(d, (r"$Q_N$", 'WB', r"$N=5$", 'final'),     edict[225])
add(d, (r"$Q_N$", 'WB', r"$N=5$", r"$\lambda(0.25)$"), edict[228])
add(d, (r"$Q_N$", 'WB', r"$N=5$", r"$\lambda(0.5)$"),  edict[231])
add(d, (r"$Q_N$", 'WB', r"$N=5$", r"$\lambda(0.75)$"), edict[234])
add(d, (r"$Q_N$", 'WB', r"$N=5$", r"$\lambda(0.9)$"),  edict[237])

add(d, (r"$Q_N$", 'WB', r"$N=10$", 'mean'),      edict[243])
add(d, (r"$Q_N$", 'WB', r"$N=10$", 'final'),     edict[246])
add(d, (r"$Q_N$", 'WB', r"$N=10$", r"$\lambda(0.25)$"), edict[249])
add(d, (r"$Q_N$", 'WB', r"$N=10$", r"$\lambda(0.5)$"),  edict[252])
add(d, (r"$Q_N$", 'WB', r"$N=10$", r"$\lambda(0.75)$"), edict[255])
add(d, (r"$Q_N$", 'WB', r"$N=10$", r"$\lambda(0.9)$"),  edict[258])

add(d, (r"$Q_N$", 'WB', r"$N=20$", 'mean'),      edict[264])
add(d, (r"$Q_N$", 'WB', r"$N=20$", 'final'),     edict[267])
add(d, (r"$Q_N$", 'WB', r"$N=20$", r"$\lambda(0.25)$"), edict[270])
add(d, (r"$Q_N$", 'WB', r"$N=20$", r"$\lambda(0.5)$"),  edict[273])
add(d, (r"$Q_N$", 'WB', r"$N=20$", r"$\lambda(0.75)$"), edict[276])
add(d, (r"$Q_N$", 'WB', r"$N=20$", r"$\lambda(0.9)$"),  edict[279])


# RNN Q (eligibility trace)
add(d, (r"$Q_{RNN}$", 'ET', r"$N=5$", 'mean'),      edict[739])
add(d, (r"$Q_{RNN}$", 'ET', r"$N=5$", 'final'),     edict[742])
add(d, (r"$Q_{RNN}$", 'ET', r"$N=5$", r"$\lambda(0.25)$"), edict[745])
add(d, (r"$Q_{RNN}$", 'ET', r"$N=5$", r"$\lambda(0.5)$"),  edict[748])
add(d, (r"$Q_{RNN}$", 'ET', r"$N=5$", r"$\lambda(0.75)$"), edict[751])
add(d, (r"$Q_{RNN}$", 'ET', r"$N=5$", r"$\lambda(0.9)$"),  edict[754])

add(d, (r"$Q_{RNN}$", 'ET', r"$N=10$", 'mean'),      edict[757])
add(d, (r"$Q_{RNN}$", 'ET', r"$N=10$", 'final'),     edict[760])
add(d, (r"$Q_{RNN}$", 'ET', r"$N=10$", r"$\lambda(0.25)$"), edict[763])
add(d, (r"$Q_{RNN}$", 'ET', r"$N=10$", r"$\lambda(0.5)$"),  edict[766])
add(d, (r"$Q_{RNN}$", 'ET', r"$N=10$", r"$\lambda(0.75)$"), edict[769])
add(d, (r"$Q_{RNN}$", 'ET', r"$N=10$", r"$\lambda(0.9)$"),  edict[772])

add(d, (r"$Q_{RNN}$", 'ET', r"$N=20$", 'mean'),      edict[775])
add(d, (r"$Q_{RNN}$", 'ET', r"$N=20$", 'final'),     edict[778])
add(d, (r"$Q_{RNN}$", 'ET', r"$N=20$", r"$\lambda(0.25)$"), edict[781])
add(d, (r"$Q_{RNN}$", 'ET', r"$N=20$", r"$\lambda(0.5)$"),  edict[784])
add(d, (r"$Q_{RNN}$", 'ET', r"$N=20$", r"$\lambda(0.75)$"), edict[787])
add(d, (r"$Q_{RNN}$", 'ET', r"$N=20$", r"$\lambda(0.9)$"),  edict[790])

# RNN Q (weighted bellman)
add(d, (r"$Q_{RNN}$", 'WB', r"$N=5$", 'mean'),      edict[621])
add(d, (r"$Q_{RNN}$", 'WB', r"$N=5$", 'final'),     edict[624])
add(d, (r"$Q_{RNN}$", 'WB', r"$N=5$", r"$\lambda(0.25)$"), edict[627])
add(d, (r"$Q_{RNN}$", 'WB', r"$N=5$", r"$\lambda(0.5)$"),  edict[630])
add(d, (r"$Q_{RNN}$", 'WB', r"$N=5$", r"$\lambda(0.75)$"), edict[633])
add(d, (r"$Q_{RNN}$", 'WB', r"$N=5$", r"$\lambda(0.9)$"),  edict[636])

add(d, (r"$Q_{RNN}$", 'WB', r"$N=10$", 'mean'),      edict[642])
add(d, (r"$Q_{RNN}$", 'WB', r"$N=10$", 'final'),     edict[645])
add(d, (r"$Q_{RNN}$", 'WB', r"$N=10$", r"$\lambda(0.25)$"), edict[648])
add(d, (r"$Q_{RNN}$", 'WB', r"$N=10$", r"$\lambda(0.5)$"),  edict[651])
add(d, (r"$Q_{RNN}$", 'WB', r"$N=10$", r"$\lambda(0.75)$"), edict[654])
add(d, (r"$Q_{RNN}$", 'WB', r"$N=10$", r"$\lambda(0.9)$"),  edict[657])

add(d, (r"$Q_{RNN}$", 'WB', r"$N=20$", 'mean'),      edict[663])
add(d, (r"$Q_{RNN}$", 'WB', r"$N=20$", 'final'),     edict[666])
add(d, (r"$Q_{RNN}$", 'WB', r"$N=20$", r"$\lambda(0.25)$"), edict[669])
add(d, (r"$Q_{RNN}$", 'WB', r"$N=20$", r"$\lambda(0.5)$"),  edict[672])
add(d, (r"$Q_{RNN}$", 'WB', r"$N=20$", r"$\lambda(0.75)$"), edict[675])
add(d, (r"$Q_{RNN}$", 'WB', r"$N=20$", r"$\lambda(0.9)$"),  edict[678])


# model based
add(d, ('MB', '', r"$N=5$", 'mean'),      edict[390])
add(d, ('MB', '', r"$N=5$", 'final'),     edict[393])
add(d, ('MB', '', r"$N=5$", r"$\lambda(0.25)$"), edict[396])
add(d, ('MB', '', r"$N=5$", r"$\lambda(0.5)$"),  edict[399])
add(d, ('MB', '', r"$N=5$", r"$\lambda(0.75)$"), edict[402])
add(d, ('MB', '', r"$N=5$", r"$\lambda(0.9)$"),  edict[405])

add(d, ('MB', '', r"$N=10$", 'mean'),      edict[411])
add(d, ('MB', '', r"$N=10$", 'final'),     edict[414])
add(d, ('MB', '', r"$N=10$", r"$\lambda(0.25)$"), edict[417])
add(d, ('MB', '', r"$N=10$", r"$\lambda(0.5)$"),  edict[420])
add(d, ('MB', '', r"$N=10$", r"$\lambda(0.75)$"), edict[423])
add(d, ('MB', '', r"$N=10$", r"$\lambda(0.9)$"),  edict[426])

add(d, ('MB', '', r"$N=20$", 'mean'),      edict[432])
add(d, ('MB', '', r"$N=20$", 'final'),     edict[435])
add(d, ('MB', '', r"$N=20$", r"$\lambda(0.25)$"), edict[438])
add(d, ('MB', '', r"$N=20$", r"$\lambda(0.5)$"),  edict[441])
add(d, ('MB', '', r"$N=20$", r"$\lambda(0.75)$"), edict[444])
add(d, ('MB', '', r"$N=20$", r"$\lambda(0.9)$"),  edict[447])


# MAC (eligibility trace)
add(d, ('MAC', 'ET', r"$N=5$", 'mean'),      edict[793])
add(d, ('MAC', 'ET', r"$N=5$", 'final'),     edict[796])
add(d, ('MAC', 'ET', r"$N=5$", r"$\lambda(0.25)$"), edict[799])
add(d, ('MAC', 'ET', r"$N=5$", r"$\lambda(0.5)$"),  edict[802])
add(d, ('MAC', 'ET', r"$N=5$", r"$\lambda(0.75)$"), edict[805])
add(d, ('MAC', 'ET', r"$N=5$", r"$\lambda(0.9)$"),  edict[808])

add(d, ('MAC', 'ET', r"$N=10$", 'mean'),      edict[811])
add(d, ('MAC', 'ET', r"$N=10$", 'final'),     edict[814])
add(d, ('MAC', 'ET', r"$N=10$", r"$\lambda(0.25)$"), edict[817])
add(d, ('MAC', 'ET', r"$N=10$", r"$\lambda(0.5)$"),  edict[820])
add(d, ('MAC', 'ET', r"$N=10$", r"$\lambda(0.75)$"), edict[823])
add(d, ('MAC', 'ET', r"$N=10$", r"$\lambda(0.9)$"),  edict[826])

add(d, ('MAC', 'ET', r"$N=20$", 'mean'),      edict[829])
add(d, ('MAC', 'ET', r"$N=20$", 'final'),     edict[832])
add(d, ('MAC', 'ET', r"$N=20$", r"$\lambda(0.25)$"), edict[835])
add(d, ('MAC', 'ET', r"$N=20$", r"$\lambda(0.5)$"),  edict[838])
add(d, ('MAC', 'ET', r"$N=20$", r"$\lambda(0.75)$"), edict[841])
add(d, ('MAC', 'ET', r"$N=20$", r"$\lambda(0.9)$"),  edict[844])

# MAC (weighted bellman)
add(d, ('MAC', 'WB', r"$N=5$", 'mean'),      edict[558])
add(d, ('MAC', 'WB', r"$N=5$", 'final'),     edict[561])
add(d, ('MAC', 'WB', r"$N=5$", r"$\lambda(0.25)$"), edict[564])
add(d, ('MAC', 'WB', r"$N=5$", r"$\lambda(0.5)$"),  edict[567])
add(d, ('MAC', 'WB', r"$N=5$", r"$\lambda(0.75)$"), edict[570])
add(d, ('MAC', 'WB', r"$N=5$", r"$\lambda(0.9)$"),  edict[573])

add(d, ('MAC', 'WB', r"$N=10$", 'mean'),      edict[579])
add(d, ('MAC', 'WB', r"$N=10$", 'final'),     edict[582])
add(d, ('MAC', 'WB', r"$N=10$", r"$\lambda(0.25)$"), edict[585])
add(d, ('MAC', 'WB', r"$N=10$", r"$\lambda(0.5)$"),  edict[588])
add(d, ('MAC', 'WB', r"$N=10$", r"$\lambda(0.75)$"), edict[591])
add(d, ('MAC', 'WB', r"$N=10$", r"$\lambda(0.9)$"),  edict[594])

add(d, ('MAC', 'WB', r"$N=20$", 'mean'),      edict[600])
add(d, ('MAC', 'WB', r"$N=20$", 'final'),     edict[603])
add(d, ('MAC', 'WB', r"$N=20$", r"$\lambda(0.25)$"), edict[606])
add(d, ('MAC', 'WB', r"$N=20$", r"$\lambda(0.5)$"),  edict[609])
add(d, ('MAC', 'WB', r"$N=20$", r"$\lambda(0.75)$"), edict[612])
add(d, ('MAC', 'WB', r"$N=20$", r"$\lambda(0.9)$"),  edict[615])

def mk_groups(data):
    try:
        newdata = data.items()
    except:
        return

    thisgroup = []
    groups = []
    for key, value in newdata:
        newgroups = mk_groups(value)
        if type(key) is str:
            key = '{0: <20}'.format(key)
        if type(value) is str:
            value = '{0: <20}'.format(value)
        if newgroups is None:
            thisgroup.append((key, value))
        else:
            thisgroup.append((key, len(newgroups[-1])))
            if groups:
                groups = [g + n for n, g in zip(newgroups, groups)]
            else:
                groups = newgroups
    return [thisgroup] + groups

def add_line(ax, xpos, ypos):
    line = plt.Line2D([xpos, xpos], [ypos + .1, ypos],
                      transform=ax.transAxes, color='black')
    line.set_clip_on(False)
    ax.add_line(line)

def label_group_bar(ax, data, ypos_shifts):
    groups = mk_groups(data)
    xy = groups.pop()
    x, y = zip(*xy)
    ly = len(y)
    xticks = range(1, ly + 1)

    ax.bar(xticks, y, align='center')
    ax.set_xticks(xticks)
    ax.set_xticklabels(x, rotation=90)
    ax.set_xlim(.5, ly + .5)
    ax.yaxis.grid(True)

    scale = 1. / ly
    # for pos in range(ly + 1):
    #    add_line(ax, pos * scale, -.1)
    ypos = -ypos_shifts.pop()
    while groups:
        group = groups.pop()
        pos = 0
        maxlen_label = -np.inf
        for label, rpos in group:
            lxpos = (pos + .5 * rpos) * scale
            # lxpos = (pos + 0.5) * scale
            ax.text(lxpos, ypos, label, ha='center', transform=ax.transAxes, rotation=90)
            # add_line(ax, pos * scale, ypos)
            pos += rpos
            maxlen_label = max(maxlen_label, len(label))
        # add_line(ax, pos * scale, ypos)
        ypos -= ypos_shifts.pop()
        # ypos -= .03 * maxlen_label

plt.rc('text', usetex=True)
plt.rc('font', family='serif')

f, ax = plt.subplots(1, 1, figsize=(20, 5))
label_group_bar(ax, d, [0.1, 0.2, 0.2, 0.3])

ax.set_ylabel('Steps until solved')
yfmt = ticker.ScalarFormatter()
yfmt.set_powerlimits((0, 0))
ax.yaxis.set_major_formatter(yfmt)
for tick in ax.get_yticklabels():
    tick.set_rotation(90)

plt.tight_layout()
f.savefig(os.path.join(SAVE_FOLDER, 'pend0_comparison.png'), bbox_inches='tight', dpi=200)
plt.close(f)


import IPython; IPython.embed()


