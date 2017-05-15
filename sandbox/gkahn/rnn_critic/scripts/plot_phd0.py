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

def process_experiments(start_index, repeat, window=10):
    analyze_names = []
    successes = []
    for index in range(start_index, start_index + repeat):
        try:
            analyze = AnalyzeRNNCritic(os.path.join(EXP_FOLDER, 'phd{0:03d}'.format(index)), clear_obs=False, create_new_envs=False, load_rollouts=False)
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

        # rollouts = list(itertools.chain(*analyze.eval_rollouts_itrs))
        # rollouts = sorted(rollouts, key=lambda r: r['steps'][0])
        # steps = [r['steps'][0] for r in rollouts]
        # values = [len(r['rewards']) for r in rollouts]

        if len(analyze_names) == 0:
            raise Exception

        length = int(analyze.params['alg']['env'].split('length=')[1].split(',')[0])
        successes.append(list(analyze.progress['EvalEpisodeLengthMean'])[-1] > length)

    return analyze_name, np.mean(successes)

names, pct_successes = [], []
for i in range(321, 782, 3):
    try:
        name, pct_success = process_experiments(i, 3, window=10)
        names.append(name)
        pct_successes.append(pct_success)
        print('Loaded {0}'.format(i))
    except:
        print('Failed to load {0}'.format(i))
        names.append('')
        pct_successes.append(np.nan)

import IPython; IPython.embed()

s = ''
for name, pct_success in zip(names, pct_successes):
    s += '{0:.1f}\t{1}\n'.format(pct_success, name)
# print(s)

with open(os.path.join(SAVE_FOLDER, 'plot_phd0.txt'), 'w') as f:
    f.write(s)
