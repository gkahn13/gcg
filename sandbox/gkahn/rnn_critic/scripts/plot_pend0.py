import os
import numpy as np
import itertools

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
        analyze = AnalyzeRNNCritic(os.path.join(EXP_FOLDER, 'pend{0:03d}'.format(index)), clear_obs=False, create_new_envs=False)

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

    return analyze_name, above_threshold_step

names, thresholds = [], []
for i in list(range(201, 683, 3)) + list(range(685, 828, 3)):
    try:
        name, threshold = process_experiments(i, 3, window=10)
        names.append(name)
        thresholds.append(threshold)
    except:
        print('Failed to load {0}'.format(i))

s = ''
for name, threshold in zip(names, thresholds):
    s += '{0:.1f}e3\t{1}\n'.format(threshold/1e3, name)
print(s)

with open(os.path.join(SAVE_FOLDER, 'plot_pend0.txt'), 'w') as f:
    f.write(s)

import IPython; IPython.embed()
