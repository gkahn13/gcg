import os, pickle
import numpy as np
import itertools

import matplotlib.pyplot as plt
from matplotlib import ticker
import matplotlib.cm as cm

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

def plot_cumreward(ax, analyze_group, color='k', label=None, window=20, success_cumreward=None):
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

        # ax.plot(steps, values, color='r', alpha=np.linspace(1., 0.4, len(analyze_group))[i])

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

    if success_cumreward is not None:
        if values_mean.max() >= success_cumreward:
            thresh_step = steps[(values_mean >= success_cumreward).argmax()]
            ax.vlines(thresh_step, (values_mean - values_std).min(), (values_mean + values_std).max(), color='g', linestyle='--')

    ax.set_xticks(np.arange(0, max(steps), 1e3), minor=True)
    ax.grid(which='minor', alpha=0.2)
    ax.grid(which='major', alpha=0.5)
    xfmt = ticker.ScalarFormatter()
    xfmt.set_powerlimits((0, 0))
    ax.xaxis.set_major_formatter(xfmt)
    ax.set_ylim((10, 60))

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

def plot_paths(axes, analyze_group):

    def get_rollout_positions(exp):
        start_steps = []
        positions = []
        for rollout in itertools.chain(*exp.eval_rollouts_itrs):
            start_steps.append(rollout['steps'][0])

            p = np.array([d['pos'][:2] for d in rollout['env_infos']])
            positions.append(p)

        return start_steps, positions

    group_start_steps, group_positions = zip(*[get_rollout_positions(exp) for exp in analyze_group])
    colors = [cm.rainbow(i) for i in np.linspace(0, 1, len(group_positions))]

    for start_steps, positions, color in zip(group_start_steps, group_positions, colors):
        start_steps_list = np.array_split(start_steps, len(axes))
        positions_list = np.array_split(positions, len(axes))
        for ax, start_steps_i, positions_i in zip(axes, start_steps_list, positions_list):
            ax.set_title('[{0:.2e}, {1:.2e}]'.format(min(start_steps_i), max(start_steps_i)),
                         fontdict={'fontsize': 6})

            for pos in positions_i[::len(positions_i)//10]:
                ax.plot(pos[:, 0], pos[:, 1], color=color, linewidth=1.)
                if len(pos) < 23:
                    ax.plot([pos[-1, 0]], [pos[-1, 1]], color=color, marker='d')

            ax.set_xlim([-4.5, 4.5])
            ax.set_ylim([-7.5, 7.5])

############################
### Specific experiments ###
############################

def plot_554_590():
    FILE_NAME = 'rccar_554_590'
    SAVE_DISTANCE = False
    SAVE_COLL = False
    SAVE_VALUE = False

    all_exps = [load_experiments(range(i, i + 3)) for i in list(range(554, 562, 3)) + list(range(564, 590, 3))]

    probcoll_exp = load_probcoll_experiments('/home/gkahn/code/probcoll/experiments/sim_rccar/test/analysis_images')

    f_cumreward, axes_cumreward = plt.subplots(4, 3, figsize=(15, 20), sharey=False, sharex=True)
    f_distance, axes_distance = plt.subplots(4, 3, figsize=(15, 20), sharey=True, sharex=True)
    f_coll, axes_coll = plt.subplots(4, 3, figsize=(15, 20), sharey=True, sharex=True)
    f_value, axes_value = plt.subplots(4, 3, figsize=(15, 20), sharey=False, sharex=True)

    for ax_cumreward, ax_distance, ax_coll, ax_value, exp in \
            zip(axes_cumreward.ravel(), axes_distance.ravel(), axes_coll.ravel(), axes_value.ravel(), all_exps):

        if not hasattr(exp, '__len__'):
            exp = [exp]

        if len(exp) > 0:
            try:
                plot_cumreward(ax_cumreward, exp, window=8, success_cumreward=40.)
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

    for all_axes in (axes_cumreward, axes_value):
        for row in range(4):
            axes = all_axes[row, :]
            ymin, ymax = np.inf, -np.inf
            for ax in axes:
                ymin = min(ymin, ax.get_ylim()[0])
                ymax = max(ymax, ax.get_ylim()[1])
            for ax in axes:
                ax.set_ylim((ymin, ymax))

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

def plot_592_627():
    FILE_NAME = 'rccar_592_627'
    SAVE_DISTANCE = False
    SAVE_COLL = False
    SAVE_VALUE = False

    all_exps = [load_experiments(range(i, i + 3)) for i in range(592, 627, 3)]

    probcoll_exp = load_probcoll_experiments('/home/gkahn/code/probcoll/experiments/sim_rccar/test/analysis_images')

    f_cumreward, axes_cumreward = plt.subplots(4, 3, figsize=(15, 20), sharey=False, sharex=True)
    f_distance, axes_distance = plt.subplots(4, 3, figsize=(15, 20), sharey=True, sharex=True)
    f_coll, axes_coll = plt.subplots(4, 3, figsize=(15, 20), sharey=True, sharex=True)
    f_value, axes_value = plt.subplots(4, 3, figsize=(15, 20), sharey=False, sharex=True)

    for ax_cumreward, ax_distance, ax_coll, ax_value, exp in \
            zip(axes_cumreward.ravel(), axes_distance.ravel(), axes_coll.ravel(), axes_value.ravel(), all_exps):

        if not hasattr(exp, '__len__'):
            exp = [exp]

        if len(exp) > 0:
            try:
                plot_cumreward(ax_cumreward, exp, window=8, success_cumreward=40.)
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

    for all_axes in (axes_cumreward, axes_value):
        for row in range(4):
            axes = all_axes[row, :]
            ymin, ymax = np.inf, -np.inf
            for ax in axes:
                ymin = min(ymin, ax.get_ylim()[0])
                ymax = max(ymax, ax.get_ylim()[1])
            for ax in axes:
                ax.set_ylim((ymin, ymax))

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

def plot_629_664():
    FILE_NAME = 'rccar_629_664'
    SAVE_DISTANCE = False
    SAVE_COLL = False
    SAVE_VALUE = False

    all_exps = [load_experiments(range(i, i + 3)) for i in range(629, 664, 3)]

    probcoll_exp = load_probcoll_experiments('/home/gkahn/code/probcoll/experiments/sim_rccar/test/analysis_images')

    f_cumreward, axes_cumreward = plt.subplots(4, 3, figsize=(15, 20), sharey=False, sharex=True)
    f_distance, axes_distance = plt.subplots(4, 3, figsize=(15, 20), sharey=True, sharex=True)
    f_coll, axes_coll = plt.subplots(4, 3, figsize=(15, 20), sharey=True, sharex=True)
    f_value, axes_value = plt.subplots(4, 3, figsize=(15, 20), sharey=False, sharex=True)

    for ax_cumreward, ax_distance, ax_coll, ax_value, exp in \
            zip(axes_cumreward.ravel(), axes_distance.ravel(), axes_coll.ravel(), axes_value.ravel(), all_exps):

        if not hasattr(exp, '__len__'):
            exp = [exp]

        if len(exp) > 0:
            try:
                plot_cumreward(ax_cumreward, exp, window=8, success_cumreward=40.)
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

    for all_axes in (axes_cumreward, axes_value):
        for row in range(4):
            axes = all_axes[row, :]
            ymin, ymax = np.inf, -np.inf
            for ax in axes:
                ymin = min(ymin, ax.get_ylim()[0])
                ymax = max(ymax, ax.get_ylim()[1])
            for ax in axes:
                ax.set_ylim((ymin, ymax))

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

def plot_666_701():
    FILE_NAME = 'rccar_666_701'
    SAVE_DISTANCE = False
    SAVE_COLL = False
    SAVE_VALUE = False

    all_exps = [load_experiments(range(i, i + 3)) for i in range(666, 701, 3)]

    probcoll_exp = load_probcoll_experiments('/home/gkahn/code/probcoll/experiments/sim_rccar/test/analysis_images')

    f_cumreward, axes_cumreward = plt.subplots(4, 3, figsize=(15, 20), sharey=False, sharex=True)
    f_distance, axes_distance = plt.subplots(4, 3, figsize=(15, 20), sharey=True, sharex=True)
    f_coll, axes_coll = plt.subplots(4, 3, figsize=(15, 20), sharey=True, sharex=True)
    f_value, axes_value = plt.subplots(4, 3, figsize=(15, 20), sharey=False, sharex=True)

    for ax_cumreward, ax_distance, ax_coll, ax_value, exp in \
            zip(axes_cumreward.ravel(), axes_distance.ravel(), axes_coll.ravel(), axes_value.ravel(), all_exps):

        if not hasattr(exp, '__len__'):
            exp = [exp]

        if len(exp) > 0:
            try:
                plot_cumreward(ax_cumreward, exp, window=8, success_cumreward=40.)
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

    for all_axes in (axes_cumreward, axes_value):
        for row in range(4):
            axes = all_axes[row, :]
            ymin, ymax = np.inf, -np.inf
            for ax in axes:
                ymin = min(ymin, ax.get_ylim()[0])
                ymax = max(ymax, ax.get_ylim()[1])
            for ax in axes:
                ax.set_ylim((ymin, ymax))

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

def plot_703_714():
    FILE_NAME = 'rccar_703_714'
    SAVE_DISTANCE = False
    SAVE_COLL = False
    SAVE_VALUE = False

    all_exps = [load_experiments(range(i, i + 3)) for i in range(703, 714, 3)]

    probcoll_exp = load_probcoll_experiments('/home/gkahn/code/probcoll/experiments/sim_rccar/test/analysis_images')

    f_cumreward, axes_cumreward = plt.subplots(1, 4, figsize=(20, 5), sharey=True, sharex=True)
    f_distance, axes_distance = plt.subplots(1, 4, figsize=(20, 5), sharey=True, sharex=True)
    f_coll, axes_coll = plt.subplots(1, 4, figsize=(20, 5), sharey=True, sharex=True)
    f_value, axes_value = plt.subplots(1, 4, figsize=(20, 5), sharey=False, sharex=True)

    for ax_cumreward, ax_distance, ax_coll, ax_value, exp in \
            zip(axes_cumreward.ravel(), axes_distance.ravel(), axes_coll.ravel(), axes_value.ravel(), all_exps):

        if not hasattr(exp, '__len__'):
            exp = [exp]

        if len(exp) > 0:
            try:
                plot_cumreward(ax_cumreward, exp, window=8, success_cumreward=40.)
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

def plot_716_727():
    FILE_NAME = 'rccar_716_727'
    SAVE_DISTANCE = False
    SAVE_COLL = False
    SAVE_VALUE = False

    all_exps = [load_experiments(range(i, i + 3)) for i in range(716, 727, 3)]

    probcoll_exp = load_probcoll_experiments('/home/gkahn/code/probcoll/experiments/sim_rccar/test/analysis_images')

    f_cumreward, axes_cumreward = plt.subplots(1, 4, figsize=(20, 5), sharey=True, sharex=True)
    f_distance, axes_distance = plt.subplots(1, 4, figsize=(20, 5), sharey=True, sharex=True)
    f_coll, axes_coll = plt.subplots(1, 4, figsize=(20, 5), sharey=True, sharex=True)
    f_value, axes_value = plt.subplots(1, 4, figsize=(20, 5), sharey=False, sharex=True)

    for ax_cumreward, ax_distance, ax_coll, ax_value, exp in \
            zip(axes_cumreward.ravel(), axes_distance.ravel(), axes_coll.ravel(), axes_value.ravel(), all_exps):

        if not hasattr(exp, '__len__'):
            exp = [exp]

        if len(exp) > 0:
            try:
                plot_cumreward(ax_cumreward, exp, window=8, success_cumreward=40.)
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

def plot_729_740():
    FILE_NAME = 'rccar_729_740'
    SAVE_DISTANCE = False
    SAVE_COLL = False
    SAVE_VALUE = False

    all_exps = [load_experiments(range(i, i + 3)) for i in range(729, 740, 3)]

    probcoll_exp = load_probcoll_experiments('/home/gkahn/code/probcoll/experiments/sim_rccar/test/analysis_images')

    f_cumreward, axes_cumreward = plt.subplots(1, 4, figsize=(20, 5), sharey=True, sharex=True)
    f_distance, axes_distance = plt.subplots(1, 4, figsize=(20, 5), sharey=True, sharex=True)
    f_coll, axes_coll = plt.subplots(1, 4, figsize=(20, 5), sharey=True, sharex=True)
    f_value, axes_value = plt.subplots(1, 4, figsize=(20, 5), sharey=False, sharex=True)

    for ax_cumreward, ax_distance, ax_coll, ax_value, exp in \
            zip(axes_cumreward.ravel(), axes_distance.ravel(), axes_coll.ravel(), axes_value.ravel(), all_exps):

        if not hasattr(exp, '__len__'):
            exp = [exp]

        if len(exp) > 0:
            try:
                plot_cumreward(ax_cumreward, exp, window=8, success_cumreward=40.)
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

def plot_742_753():
    FILE_NAME = 'rccar_742_753'
    SAVE_DISTANCE = False
    SAVE_COLL = False
    SAVE_VALUE = False

    all_exps = [load_experiments(range(i, i + 3)) for i in range(742, 753, 3)]

    probcoll_exp = load_probcoll_experiments('/home/gkahn/code/probcoll/experiments/sim_rccar/test/analysis_images')

    f_cumreward, axes_cumreward = plt.subplots(1, 4, figsize=(20, 5), sharey=True, sharex=True)
    f_distance, axes_distance = plt.subplots(1, 4, figsize=(20, 5), sharey=True, sharex=True)
    f_coll, axes_coll = plt.subplots(1, 4, figsize=(20, 5), sharey=True, sharex=True)
    f_value, axes_value = plt.subplots(1, 4, figsize=(20, 5), sharey=False, sharex=True)

    for ax_cumreward, ax_distance, ax_coll, ax_value, exp in \
            zip(axes_cumreward.ravel(), axes_distance.ravel(), axes_coll.ravel(), axes_value.ravel(), all_exps):

        if not hasattr(exp, '__len__'):
            exp = [exp]

        if len(exp) > 0:
            try:
                plot_cumreward(ax_cumreward, exp, window=8, success_cumreward=40.)
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

def plot_755_766():
    FILE_NAME = 'rccar_755_766'
    SAVE_DISTANCE = False
    SAVE_COLL = False
    SAVE_VALUE = False

    all_exps = [load_experiments(range(i, i + 3)) for i in range(755, 766, 3)]

    probcoll_exp = load_probcoll_experiments('/home/gkahn/code/probcoll/experiments/sim_rccar/test/analysis_images')

    f_cumreward, axes_cumreward = plt.subplots(1, 4, figsize=(20, 5), sharey=True, sharex=True)
    f_distance, axes_distance = plt.subplots(1, 4, figsize=(20, 5), sharey=True, sharex=True)
    f_coll, axes_coll = plt.subplots(1, 4, figsize=(20, 5), sharey=True, sharex=True)
    f_value, axes_value = plt.subplots(1, 4, figsize=(20, 5), sharey=False, sharex=True)

    for ax_cumreward, ax_distance, ax_coll, ax_value, exp in \
            zip(axes_cumreward.ravel(), axes_distance.ravel(), axes_coll.ravel(), axes_value.ravel(), all_exps):

        if not hasattr(exp, '__len__'):
            exp = [exp]

        if len(exp) > 0:
            try:
                plot_cumreward(ax_cumreward, exp, window=8, success_cumreward=40.)
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
                        params['alg']['env_eval'].split(':')[-1].split('}')[0]
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

def plot_768_911():
    FILE_NAME = 'rccar_768_911'

    all_exps = np.array([load_experiments(range(i, i + 3)) for i in range(768, 911, 3)])

    probcoll_exp = load_probcoll_experiments('/home/gkahn/code/probcoll/experiments/sim_rccar/test/analysis_images')

    for i, exps in enumerate(np.split(all_exps, 3)):

        f_cumreward, axes_cumreward = plt.subplots(4, 4, figsize=(15, 15), sharey=True, sharex=True)

        for ax_cumreward, exp in zip(axes_cumreward.ravel(), exps):

            if not hasattr(exp, '__len__'):
                exp = [exp]

            if len(exp) > 0:
                try:
                    plot_cumreward(ax_cumreward, exp, window=8, success_cumreward=40.)
                    if probcoll_exp is not None:
                        plot_cumreward_probcoll(ax_cumreward, probcoll_exp)
                    params = exp[0].params
                    policy = params['policy'][params['policy']['class']]
                    for ax in (ax_cumreward,):
                        ax.set_title('{0}, {1}, is_classif: {2},\ntarg: {3}, rp: {4}, N: {5}, H: {6}'.format(
                            params['exp_name'],
                            params['policy']['class'],
                            params['policy']['RCcarMACPolicy']['is_classification'],
                            params['policy']['use_target'],
                            params['alg']['replay_pool_sampling'],
                            params['policy']['N'],
                            params['policy']['H'],
                        ), fontdict={'fontsize': 8})
                except:
                    pass

        f_cumreward.savefig(os.path.join(SAVE_FOLDER, '{0}_cumreward_{1}.png'.format(FILE_NAME, i)),
                            bbox_inches='tight', dpi=150)

        plt.close(f_cumreward)

def plot_913_930():
    FILE_NAME = 'rccar_913_930'
    SAVE_DISTANCE = False
    SAVE_COLL = False
    SAVE_VALUE = False

    all_exps = [load_experiments(range(i, i + 3)) for i in list(range(913, 921, 3)) + list(range(923, 930, 3))]

    probcoll_exp = load_probcoll_experiments('/home/gkahn/code/probcoll/experiments/sim_rccar/test/analysis_images')

    f_cumreward, axes_cumreward = plt.subplots(2, 3, figsize=(15, 10), sharey=True, sharex=True)
    f_distance, axes_distance = plt.subplots(2, 3, figsize=(15, 10), sharey=True, sharex=True)
    f_coll, axes_coll = plt.subplots(2, 3, figsize=(15, 10), sharey=True, sharex=True)
    f_value, axes_value = plt.subplots(2, 3, figsize=(15, 10), sharey=False, sharex=True)

    for ax_cumreward, ax_distance, ax_coll, ax_value, exp in \
            zip(axes_cumreward.ravel(), axes_distance.ravel(), axes_coll.ravel(), axes_value.ravel(), all_exps):

        if not hasattr(exp, '__len__'):
            exp = [exp]

        if len(exp) > 0:
            try:
                plot_cumreward(ax_cumreward, exp, window=8, success_cumreward=40.)
                if probcoll_exp is not None:
                    plot_cumreward_probcoll(ax_cumreward, probcoll_exp)
                plot_distance(ax_distance, exp, window=20)
                plot_collisions(ax_coll, exp)
                plot_value(ax_value, exp, window=4)
                params = exp[0].params
                policy = params['policy'][params['policy']['class']]
                for ax in (ax_cumreward, ax_distance, ax_coll, ax_value):
                    ax.set_title('{0}, N: {1}, H: {2}, speeds: {3}'.format(
                        params['policy']['class'],
                        params['policy']['N'],
                        params['policy']['H'],
                        params['alg']['env_eval'].split("'speed_limits':")[-1].split('}')[0]
                    ), fontdict={'fontsize': 6})
                ax_distance.set_ylim((0, 15))
            except:
                pass

    f_cumreward.savefig(os.path.join(SAVE_FOLDER, '{0}_cumreward.png'.format(FILE_NAME)), bbox_inches='tight',
                        dpi=150)
    if SAVE_DISTANCE:
        f_distance.savefig(os.path.join(SAVE_FOLDER, '{0}_distance.png'.format(FILE_NAME)), bbox_inches='tight',
                           dpi=150)
    if SAVE_COLL:
        f_coll.savefig(os.path.join(SAVE_FOLDER, '{0}_coll.png'.format(FILE_NAME)), bbox_inches='tight', dpi=150)
    if SAVE_VALUE:
        f_value.savefig(os.path.join(SAVE_FOLDER, '{0}_value.png'.format(FILE_NAME)), bbox_inches='tight', dpi=150)

    plt.close(f_cumreward)
    plt.close(f_distance)
    plt.close(f_coll)
    plt.close(f_value)

def plot_932_949():
    FILE_NAME = 'rccar_932_949'

    all_exps = [load_experiments(range(i, i + 3)) for i in range(932, 949, 3)]

    probcoll_exp = load_probcoll_experiments('/home/gkahn/code/probcoll/experiments/sim_rccar/test/analysis_images')

    f_cumreward, axes_cumreward = plt.subplots(2, 3, figsize=(15, 10), sharey=True, sharex=True)
    paths_cols = 6
    f_paths, axes_paths = plt.subplots(len(all_exps), paths_cols, figsize=(2*paths_cols, 2*len(all_exps)), sharey=True, sharex=True)

    for ax_cumreward, axes_paths_row, exp in \
            zip(axes_cumreward.ravel(), axes_paths, all_exps):

        if not hasattr(exp, '__len__'):
            exp = [exp]

        if len(exp) > 0:
            try:
                plot_cumreward(ax_cumreward, exp, window=8, success_cumreward=40.)
                if probcoll_exp is not None:
                    plot_cumreward_probcoll(ax_cumreward, probcoll_exp)
                plot_paths(axes_paths_row, exp)
                params = exp[0].params
                policy = params['policy'][params['policy']['class']]
                for ax in (ax_cumreward,):
                    ax.set_title('{0}, N: {1}, H: {2}, speeds: {3}'.format(
                        params['policy']['class'],
                        params['policy']['N'],
                        params['policy']['H'],
                        params['alg']['env_eval'].split("'speed_limits':")[-1].split('}')[0]
                    ), fontdict={'fontsize': 6})
            except:
                pass

    f_cumreward.savefig(os.path.join(SAVE_FOLDER, '{0}_cumreward.png'.format(FILE_NAME)), bbox_inches='tight', dpi=150)
    f_paths.savefig(os.path.join(SAVE_FOLDER, '{0}_paths.png'.format(FILE_NAME)), bbox_inches='tight', dpi=150)

    plt.close(f_cumreward)
    plt.close(f_paths)

def plot_951_974():
    FILE_NAME = 'rccar_951_974'

    all_exps = [load_experiments(range(i, i + 3)) for i in range(951, 974, 3)]

    probcoll_exp = load_probcoll_experiments('/home/gkahn/code/probcoll/experiments/sim_rccar/test/analysis_images')

    f_cumreward, axes_cumreward = plt.subplots(2, 4, figsize=(20, 10), sharey=True, sharex=True)
    paths_cols = 6
    f_paths, axes_paths = plt.subplots(len(all_exps), paths_cols, figsize=(2*paths_cols, 2*len(all_exps)), sharey=True, sharex=True)

    for ax_cumreward, axes_paths_row, exp in \
            zip(axes_cumreward.ravel(), axes_paths, all_exps):

        if not hasattr(exp, '__len__'):
            exp = [exp]

        if len(exp) > 0:
            try:
                plot_cumreward(ax_cumreward, exp, window=8, success_cumreward=40.)
                if probcoll_exp is not None:
                    plot_cumreward_probcoll(ax_cumreward, probcoll_exp)
                plot_paths(axes_paths_row, exp)
                params = exp[0].params
                policy = params['policy'][params['policy']['class']]
                for ax in (ax_cumreward,):
                    ax.set_title('{0}, N: {1}, H: {2}, speeds: {3}'.format(
                        params['policy']['class'],
                        params['policy']['N'],
                        params['policy']['H'],
                        params['alg']['env_eval'].split("'speed_limits':")[-1].split('}')[0]
                    ), fontdict={'fontsize': 6})
            except:
                pass

    f_cumreward.savefig(os.path.join(SAVE_FOLDER, '{0}_cumreward.png'.format(FILE_NAME)), bbox_inches='tight', dpi=150)
    f_paths.savefig(os.path.join(SAVE_FOLDER, '{0}_paths.png'.format(FILE_NAME)), bbox_inches='tight', dpi=150)

    plt.close(f_cumreward)
    plt.close(f_paths)

def plot_976_1002():
    FILE_NAME = 'rccar_976_1002'

    all_exps = [load_experiments(range(976, 976 + 3))] + 4*[[]] + \
               [[]] + [load_experiments(range(i, i + 3)) for i in range(979, 991, 3)] + \
               [[]] + [load_experiments(range(i, i + 3)) for i in range(991, 1002, 3)]

    probcoll_exp = load_probcoll_experiments('/home/gkahn/code/probcoll/experiments/sim_rccar/test/analysis_images')

    f_cumreward, axes_cumreward = plt.subplots(3, 5, figsize=(20, 12), sharey=True, sharex=True)

    for ax_cumreward, exp in zip(axes_cumreward.ravel(), all_exps):

        if not hasattr(exp, '__len__'):
            exp = [exp]

        if len(exp) > 0:
            try:
                plot_cumreward(ax_cumreward, exp, window=8, success_cumreward=40.)
                if probcoll_exp is not None:
                    plot_cumreward_probcoll(ax_cumreward, probcoll_exp)
                params = exp[0].params
                policy = params['policy'][params['policy']['class']]
                for ax in (ax_cumreward,):
                    ax.set_title('{0}, N: {1}, H: {2}, speeds: {3}'.format(
                        params['policy']['class'],
                        params['policy']['N'],
                        params['policy']['H'],
                        params['alg']['env_eval'].split("'speed_limits':")[-1].split('}')[0]
                    ), fontdict={'fontsize': 6})
            except:
                pass

    f_cumreward.savefig(os.path.join(SAVE_FOLDER, '{0}_cumreward.png'.format(FILE_NAME)), bbox_inches='tight', dpi=150)

    plt.close(f_cumreward)

def plot_1004_1147():
    FILE_NAME = 'rccar_1004_1147'

    all_exps = np.array([load_experiments(range(i, i + 3)) for i in range(1004, 1147, 3)])

    probcoll_exp = load_probcoll_experiments('/home/gkahn/code/probcoll/experiments/sim_rccar/test/analysis_images')

    import IPython; IPython.embed()

    for i, exps in enumerate(np.split(all_exps, 3)):

        f_cumreward, axes_cumreward = plt.subplots(4, 4, figsize=(15, 15), sharey=True, sharex=True)

        for ax_cumreward, exp in zip(axes_cumreward.ravel(), exps):

            if not hasattr(exp, '__len__'):
                exp = [exp]

            if len(exp) > 0:
                try:
                    plot_cumreward(ax_cumreward, exp, window=8, success_cumreward=40.)
                    if probcoll_exp is not None:
                        plot_cumreward_probcoll(ax_cumreward, probcoll_exp)
                    params = exp[0].params
                    policy = params['policy'][params['policy']['class']]
                    for ax in (ax_cumreward,):
                        ax.set_title('{0}, {1}, targ: {2}, classif: {3},\nreweight: {4}, rp: {5}, N: {6}, H: {7}'.format(
                            params['exp_name'],
                            params['policy']['class'],
                            params['policy']['use_target'],
                            params['policy']['RCcarMACPolicy']['is_classification'] if params['policy']['class'] == 'RCcarMACPolicy' else '',
                            params['policy']['RCcarMACPolicy']['reweight_coll_nocoll'] if params['policy']['class'] == 'RCcarMACPolicy' else '',
                            params['alg']['replay_pool_sampling'],
                            params['policy']['N'],
                            params['policy']['H'],
                        ), fontdict={'fontsize': 6})
                except:
                    pass

        f_cumreward.savefig(os.path.join(SAVE_FOLDER, '{0}_cumreward_{1}.png'.format(FILE_NAME, i)),
                            bbox_inches='tight', dpi=150)

        plt.close(f_cumreward)

def plot_test():
    FILE_NAME = 'rccar_test'

    exps = [[AnalyzeRNNCritic('/home/gkahn/code/rllab/data/local/rnn-critic/test_rccar',
                     clear_obs=False,
                     create_new_envs=False,
                     load_train_rollouts=False,
                     load_eval_rollouts=True)],
            load_experiments(range(994, 994 + 3))]
    probcoll_exp = load_probcoll_experiments('/home/gkahn/code/probcoll/experiments/sim_rccar/test/analysis_images')


    ### cumreward
    f_cumreward, ax_cumreward = plt.subplots(1, 1, figsize=(20, 10), sharey=True, sharex=True)

    plot_cumreward(ax_cumreward, exps[0], window=8, success_cumreward=40., color='r')
    plot_cumreward(ax_cumreward, exps[1], window=8, success_cumreward=40., color='k')
    plot_cumreward_probcoll(ax_cumreward, probcoll_exp)

    ax_cumreward.set_xlim((0, 1e4))

    f_cumreward.savefig(os.path.join(SAVE_FOLDER, '{0}_cumreward.png'.format(FILE_NAME)), bbox_inches='tight', dpi=150)
    plt.close(f_cumreward)

    ### paths
    paths_cols = 6
    f_paths, axes_paths = plt.subplots(len(exps), paths_cols, figsize=(2*paths_cols, 2*len(exps)), sharey=True, sharex=True)
    if len(axes_paths.shape) == 1:
        axes_paths = np.array([axes_paths])

    plot_paths(axes_paths[0], exps[0])

    f_paths.savefig(os.path.join(SAVE_FOLDER, '{0}_paths.png'.format(FILE_NAME)), bbox_inches='tight', dpi=150)
    plt.close(f_paths)

def plot_1149_1154():
    FILE_NAME = 'rccar_1149_1154'

    all_exps = [load_experiments(range(i, i + 3)) for i in range(1149, 1154, 3)]

    probcoll_exp = load_probcoll_experiments('/home/gkahn/code/probcoll/experiments/sim_rccar/test/analysis_images\
')

    f_cumreward, axes_cumreward = plt.subplots(1, 2, figsize=(10, 5), sharey=True, sharex=True)

    for ax_cumreward, exp in \
            zip(axes_cumreward.ravel(), all_exps):

        if not hasattr(exp, '__len__'):
            exp = [exp]

        if len(exp) > 0:
            try:
                plot_cumreward(ax_cumreward, exp, window=8, success_cumreward=40.)
                if probcoll_exp is not None:
                    plot_cumreward_probcoll(ax_cumreward, probcoll_exp)
                params = exp[0].params
                policy = params['policy'][params['policy']['class']]
                for ax in (ax_cumreward,):
                    ax.set_title('{0}, N: {1}, H: {2}, clip: {3}'.format(
                        params['policy']['class'],
                        params['policy']['N'],
                        params['policy']['H'],
                        params['policy']['clip_cost_target_with_dones'],
                    ), fontdict={'fontsize': 6})
            except:
                pass

    f_cumreward.savefig(os.path.join(SAVE_FOLDER, '{0}_cumreward.png'.format(FILE_NAME)), bbox_inches='tight', dpi\
=150)

    plt.close(f_cumreward)



# plot_554_590()
# plot_592_627()
# plot_629_664()
# plot_666_701()
# plot_703_714()
# plot_716_727()
# plot_729_740()
# plot_742_753()
# plot_755_766()
# plot_768_911()
# plot_913_930()
# plot_932_949()
# plot_951_974()
# plot_976_1002()
# plot_1004_1147()
# plot_1149_1154()

plot_test()
