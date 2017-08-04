import os, pickle
import numpy as np
import itertools

import matplotlib.pyplot as plt
from matplotlib import ticker

from analyze_experiment import AnalyzeRNNCritic
from sandbox.gkahn.rnn_critic.utils.utils import DataAverageInterpolation

EXP_FOLDER = '/media/gkahn/ExtraDrive1/rllab/s3/rnn-critic'
SAVE_FOLDER = '/media/gkahn/ExtraDrive1/rllab/rnn_critic/final_plots'

########################
### Load experiments ###
########################

def load_experiments(indices):
    exps = []
    for i in indices:
        try:
            exps.append(AnalyzeRNNCritic(os.path.join(EXP_FOLDER, 'rccar{0:03d}'.format(i)),
                                         clear_obs=False,
                                         create_new_envs=False,
                                         load_train_rollouts=False,
                                         load_eval_rollouts=True))
            print(i)
        except:
            pass

    return exps

def load_probcoll_experiments(exp_folder):
    exp = {}
    with open(os.path.join(exp_folder, 'testing_cumreward.pkl'), 'rb') as f:
        exp.update(pickle.load(f))
    return exp

############
### Plot ###
############

def plot_cumreward(ax, analyze_group, color='k', label=None, window=20):
    data_interp = DataAverageInterpolation()
    min_step = max_step = None
    for i, analyze in enumerate(analyze_group):
        # steps = np.array(analyze.progress['Step'])
        # values = np.array(analyze.progress['EvalCumRewardMean'])

        steps = np.array([r['steps'][0] for r in itertools.chain(*analyze.eval_rollouts_itrs)])
        values = np.array([np.sum(r['rewards']) for r in itertools.chain(*analyze.eval_rollouts_itrs)])
        all_steps, all_values = zip(*sorted(zip(steps, values), key=lambda k: k[0]))
        steps, values = [], []
        for s, v in zip(all_steps, all_values):
            if len(steps) == 0 or steps[-1] != s:
                steps.append(s)
                values.append(v)

        steps, values = zip(*[(s, v) for s, v in zip(steps, values) if np.isfinite(v)])

        def moving_avg_std(idxs, data, window):
            avg_idxs, means, stds = [], [], []
            for i in range(window, len(data)):
                avg_idxs.append(np.mean(idxs[i - window:i]))
                means.append(np.mean(data[i - window:i]))
                stds.append(np.std(data[i - window:i]))
            return avg_idxs, np.asarray(means), np.asarray(stds)

        steps, values, _ = moving_avg_std(steps, values, window=window)

        ax.plot(steps, values, color='r', alpha=np.linspace(1., 0.4, len(analyze_group))[i])

        try:
            data_interp.add_data(steps, values)
        except:
            continue
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

    ax.grid()
    xfmt = ticker.ScalarFormatter()
    xfmt.set_powerlimits((0, 0))
    ax.xaxis.set_major_formatter(xfmt)

def plot_cumreward_probcoll(ax, exp, color='b'):
    steps = exp['steps']
    cumrewards_mean = np.array(exp['cumrewards_mean'])
    cumrewards_std = np.array(exp['cumrewards_std'])

    ax.plot(steps, cumrewards_mean, color=color)
    ax.fill_between(steps, cumrewards_mean + cumrewards_std, cumrewards_mean - cumrewards_std, color=color, alpha=0.4)

def plot_distance(ax, analyze_group, color='k', label=None, window=20):
    data_interp = DataAverageInterpolation()
    min_step = max_step = None
    for i, analyze in enumerate(analyze_group):
        steps = np.array([r['steps'][0] for r in itertools.chain(*analyze.eval_rollouts_itrs)])
        values = np.array([np.sum(r['rewards'][:-1] * 0.25) for r in itertools.chain(*analyze.eval_rollouts_itrs)])
        all_steps, all_values = zip(*sorted(zip(steps, values), key=lambda k: k[0]))
        steps, values = [], []
        for s, v in zip(all_steps, all_values):
            if len(steps) == 0 or steps[-1] != s:
                steps.append(s)
                values.append(v)

        steps, values = zip(*[(s, v) for s, v in zip(steps, values) if np.isfinite(v)])

        def moving_avg_std(idxs, data, window):
            avg_idxs, means, stds = [], [], []
            for i in range(window, len(data)):
                avg_idxs.append(np.mean(idxs[i - window:i]))
                means.append(np.mean(data[i - window:i]))
                stds.append(np.std(data[i - window:i]))
            return avg_idxs, np.asarray(means), np.asarray(stds)

        steps, values, _ = moving_avg_std(steps, values, window=window)

        ax.plot(steps, values, color='r', alpha=np.linspace(1., 0.4, len(analyze_group))[i])

        try:
            data_interp.add_data(steps, values)
        except:
            continue
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

    ax.grid()
    xfmt = ticker.ScalarFormatter()
    xfmt.set_powerlimits((0, 0))
    ax.xaxis.set_major_formatter(xfmt)

def plot_collisions(ax, analyze_group, color='k'):
    steps, values = [], []
    for i, analyze in enumerate(analyze_group):
        steps_i = np.array([r['steps'][0] for r in itertools.chain(*analyze.eval_rollouts_itrs)])
        values_i = np.array([len(r['rewards']) < 1e4 - 1 for r in itertools.chain(*analyze.eval_rollouts_itrs)])
        all_steps, all_values = zip(*sorted(zip(steps_i, values_i), key=lambda k: k[0]))
        for s, v in zip(all_steps, all_values):
            if len(steps) == 0 or steps[-1] != s:
                steps.append(s)
                values.append(v)

    steps, values = zip(*sorted(zip(steps, values), key=lambda k: k[0]))

    ax.plot(steps, np.cumsum(values).astype(float) / len(analyze_group), color=color)

    ax.grid()
    xfmt = ticker.ScalarFormatter()
    xfmt.set_powerlimits((0, 0))
    ax.xaxis.set_major_formatter(xfmt)

def plot_value(ax, analyze_group, window=20):
    # EstValuesMaxDiffMean, EstValuesMaxDiffStd
    # EstValuesAvgDiffMean, EstValuesAvgDiffStd
    # EstValuesMinDiffMean, EstValuesMinDiffStd

    for i, analyze in enumerate(analyze_group):
        if analyze.progress is None:
            continue
        steps = np.array(analyze.progress['Step'])
        max_values = np.array(analyze.progress['EstValuesMaxDiffMean'])
        avg_values = np.array(analyze.progress['EstValuesAvgDiffMean'])
        min_values = np.array(analyze.progress['EstValuesMinDiffMean'])

        def moving_avg_std(idxs, data, window):
            avg_idxs, means, stds = [], [], []
            for i in range(window, len(data)):
                avg_idxs.append(np.mean(idxs[i - window:i]))
                means.append(np.mean(data[i - window:i]))
                stds.append(np.std(data[i - window:i]))
            return avg_idxs, np.asarray(means), np.asarray(stds)

        steps, max_values, _ = moving_avg_std(steps, max_values, window=window)
        steps, avg_values, _ = moving_avg_std(steps, avg_values, window=window)
        steps, min_values, _ = moving_avg_std(steps, min_values, window=window)

        ax.plot(steps, max_values, color='r', alpha=np.linspace(1., 0.4, len(analyze_group))[i])
        ax.plot(steps, avg_values, color='k', alpha=np.linspace(1., 0.4, len(analyze_group))[i])
        ax.plot(steps, min_values, color='b', alpha=np.linspace(1., 0.4, len(analyze_group))[i])

    ax.grid()
    xfmt = ticker.ScalarFormatter()
    xfmt.set_powerlimits((0, 0))
    ax.xaxis.set_major_formatter(xfmt)

############################
### Specific experiments ###
############################

def plot_554_562():
    FILE_NAME = 'rccar_554_562'
    SAVE_DISTANCE = False
    SAVE_COLL = False
    SAVE_VALUE = False

    all_exps = [load_experiments(range(i, i + 3)) for i in range(554, 562, 3)]

    probcoll_exp = load_probcoll_experiments('/home/gkahn/code/probcoll/experiments/sim_rccar/test/analysis_images')

    f_cumreward, axes_cumreward = plt.subplots(1, 3, figsize=(15, 5), sharey=True, sharex=True)
    f_distance, axes_distance = plt.subplots(1, 3, figsize=(15, 5), sharey=True, sharex=True)
    f_coll, axes_coll = plt.subplots(1, 3, figsize=(15, 5), sharey=True, sharex=True)
    f_value, axes_value = plt.subplots(1, 3, figsize=(15, 5), sharey=True, sharex=True)

    for ax_cumreward, ax_distance, ax_coll, ax_value, exp in \
            zip(axes_cumreward.ravel(), axes_distance.ravel(), axes_coll.ravel(), axes_value.ravel(), all_exps):

        if not hasattr(exp, '__len__'):
            exp = [exp]

        if len(exp) > 0:
            try:
                plot_cumreward(ax_cumreward, exp, window=20)
                if probcoll_exp is not None:
                    plot_cumreward_probcoll(ax_cumreward, probcoll_exp)
                plot_distance(ax_distance, exp, window=20)
                plot_collisions(ax_coll, exp)
                plot_value(ax_value, exp, window=4)
                params = exp[0].params
                policy = params['policy'][params['policy']['class']]
                for ax in (ax_cumreward, ax_distance, ax_coll, ax_value):
                    ax.set_title('{0}, N: {1}, H: {2}, coll reward: {3}'.format(
                        params['policy']['class'],
                        params['policy']['N'],
                        params['policy']['H'],
                        params['alg']['env'].split(':')[-1].split('}')[0]
                    ), fontdict={'fontsize': 6})
                ax_distance.set_ylim((0, 15))
            except:
                pass

    f_cumreward.savefig(os.path.join(SAVE_FOLDER, '{0}_cumreward.png'.format(FILE_NAME)), bbox_inches='tight', dpi=150)
    if SAVE_DISTANCE:
        f_distance.savefig(os.path.join(SAVE_FOLDER, '{0}_distance.png'.format(FILE_NAME)), bbox_inches='tight', dpi=150)
    if SAVE_COLL:
        f_coll.savefig(os.path.join(SAVE_FOLDER, '{0}_coll.png'.format(FILE_NAME)), bbox_inches='tight', dpi=150)
    if SAVE_VALUE:
        f_value.savefig(os.path.join(SAVE_FOLDER, '{0}_value.png'.format(FILE_NAME)), bbox_inches='tight', dpi=150)

    plt.close(f_cumreward)
    plt.close(f_distance)
    plt.close(f_coll)
    plt.close(f_value)

plot_554_562()
