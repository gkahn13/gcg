import os, pickle
import numpy as np
import itertools

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import ticker
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

from analyze_experiment import AnalyzeRNNCritic
from sandbox.gkahn.rnn_critic.utils.utils import DataAverageInterpolation

from sandbox.gkahn.rnn_critic.policies.mac_policy import MACPolicy
from robots.sim_rccar.analysis.analyze_sim_rccar import AnalyzeSimRCcar

EXP_FOLDER = '/media/gkahn/ExtraDrive1/rllab/s3/rnn-critic'
SAVE_FOLDER = '/media/gkahn/ExtraDrive1/rllab/rnn_critic/final_plots'

########################
### Load experiments ###
########################

def load_experiments(indices, create_new_envs=False, load_eval_rollouts=True):
    exps = []
    for i in indices:
        try:
            exps.append(AnalyzeRNNCritic(os.path.join(EXP_FOLDER, 'rccar{0:03d}'.format(i)),
                                         clear_obs=True,
                                         create_new_envs=create_new_envs,
                                         load_train_rollouts=False,
                                         load_eval_rollouts=load_eval_rollouts))
            print(i)
        except:
            pass

    return exps

def load_probcoll_experiments(exp_folder, num):
    save_dir = os.path.join(exp_folder, 'sim_rccar{0:3d}/sim_rccar{0:3d}'.format(num))

    analyze = AnalyzeSimRCcar(save_dir=save_dir)

    steps_and_samples, _ = analyze._load_testing_samples()

    samples_per_itr = analyze.params['probcoll']['T']
    steps, samples = zip(*[(samples_per_itr * step, sample) for step, sample in steps_and_samples])
    rewards = [[s.get_X(sub_state='velocity') * (1 - s.get_X(sub_state='collision')) for s in samples_itr] for samples_itr in samples]

    # steps = np.repeat(steps, analyze.params['probcoll']['testing']['num_rollout'])
    # samples = list(itertools.chain(*samples))
    # rewards = list(itertools.chain(*rewards))

    return {'exp_num': num, 'steps': steps, 'samples': samples, 'rewards': rewards}

############
### Misc ###
############

def moving_avg_std(idxs, data, window):
    avg_idxs, means, stds = [], [], []
    for i in range(window, len(data)):
        avg_idxs.append(np.mean(idxs[i - window:i]))
        means.append(np.mean(data[i - window:i]))
        stds.append(np.std(data[i - window:i]))
    return avg_idxs, np.asarray(means), np.asarray(stds)

############
### Plot ###
############

def plot_cumreward(ax, analyze_group, color='k', label=None, window=20, success_cumreward=None, ylim=(10, 60),
                   plot_indiv=True, convert_to_time=False, xmax=None):
    data_interp = DataAverageInterpolation()
    if 'type' not in analyze_group[0].params['alg'] or analyze_group[0].params['alg']['type'] == 'interleave':
        min_step = max_step = None
        for i, analyze in enumerate(analyze_group):

            try:
                steps = np.asarray(analyze.progress['Step'], dtype=np.float32)
                values = np.asarray(analyze.progress['EvalCumRewardMean'])
                num_episodes = int(np.median(np.asarray(analyze.progress['EvalNumEpisodes'])))
                steps, values, _ = moving_avg_std(steps, values, window=window//num_episodes)

                # steps = np.array([r['steps'][0] for r in itertools.chain(*analyze.eval_rollouts_itrs)])
                # values = np.array([np.sum(r['rewards']) for r in itertools.chain(*analyze.eval_rollouts_itrs)])
                #
                # steps, values = zip(*sorted(zip(steps, values), key=lambda k: k[0]))
                # steps, values = zip(*[(s, v) for s, v in zip(steps, values) if np.isfinite(v)])
                #
                # steps, values, _ = moving_avg_std(steps, values, window=window)

                if plot_indiv:
                    assert(not convert_to_time)
                    ax.plot(steps, values, color='r', alpha=np.linspace(1., 0.4, len(analyze_group))[i])

                data_interp.add_data(steps, values)
            except:
                continue

            if min_step is None:
                min_step = steps[0]
            if max_step is None:
                max_step = steps[-1]
            min_step = max(min_step, steps[0])
            max_step = min(max_step, steps[-1])

        if len(data_interp.xs) == 0:
            return

        steps = np.r_[min_step:max_step:50.][1:-1]
        values_mean, values_std = data_interp.eval(steps)
        # steps -= min_step

    elif analyze_group[0].params['alg']['type'] == 'batch':
        all_steps = []
        values_mean = []
        values_std = []
        for i, analyze in enumerate(analyze_group):
            steps = np.array([r['steps'][0] for r in itertools.chain(*analyze.eval_rollouts_itrs)])
            values = np.array([np.sum(r['rewards']) for r in itertools.chain(*analyze.eval_rollouts_itrs)])

            eval_batch_size = analyze.params['alg']['batch']['eval_samples_per_batch']
            steps = np.reshape(steps, (-1, eval_batch_size))
            values = np.reshape(values, (-1, eval_batch_size))

            assert((steps.std(axis=1) == 0).all())

            ax.plot(steps[:, 0], values.mean(axis=1), color='r', alpha=np.linspace(1., 0.4, len(analyze_group))[i])

            all_steps.append(steps)
            values_mean.append(values.mean(axis=1))
            values_std.append(values.std(axis=1))

        min_num_batches = min([len(s) for s in all_steps])
        steps = all_steps[0][:min_num_batches, 0]
        values_mean = np.mean([v[:min_num_batches] for v in values_mean], axis=0)
        values_std = np.mean([v[:min_num_batches] for v in values_std], axis=0)

    else:
        raise NotImplementedError

    if convert_to_time:
        steps = (0.25 / 3600.) * steps

    ax.plot(steps, values_mean, color=color, label=label)
    ax.fill_between(steps, values_mean - values_std, values_mean + values_std,
                    color=color, alpha=0.4)

    xmax = xmax if xmax is not None else max(steps)
    ax.set_xticks(np.arange(0, xmax, 1e4 if not convert_to_time else 1), minor=True)
    # ax.set_xticks(np.arange(0, max(steps), 3e4 if not convert_to_time else 3), minor=False)
    # import IPython; IPython.embed()
    # ax.set_xticks(np.arange(0, max(steps), max(steps) // 18), minor=True)
    ax.set_xticks(np.arange(0, xmax, 3e4 if not convert_to_time else xmax // 4), minor=False)
    ax.grid(which='minor', alpha=0.2)
    ax.grid(which='major', alpha=0.5)
    if not convert_to_time:
        xfmt = ticker.ScalarFormatter()
        xfmt.set_powerlimits((0, 0))
        ax.xaxis.set_major_formatter(xfmt)
    if ylim is not None:
        ax.set_ylim(ylim)

    if success_cumreward is not None:
        if not hasattr(success_cumreward, '__iter__'):
            success_cumreward = [success_cumreward]

        for i, sc in enumerate(success_cumreward):
            if values_mean.max() >= sc:
                thresh_step = steps[(values_mean >= sc).argmax()]
                # color = cm.viridis(i / float(len(success_cumreward)))
                color = ['b', 'm', 'c', 'r'][i]
                ax.vlines(thresh_step, *ax.get_ylim(), color=color, linestyle='--')

def get_threshold_steps_and_final_performance(analyze_group, success_cumreward, convert_to_time=False, window=20):
    data_interp = DataAverageInterpolation()
    if 'type' not in analyze_group[0].params['alg'] or analyze_group[0].params['alg']['type'] == 'interleave':
        min_step = max_step = None
        for i, analyze in enumerate(analyze_group):

            try:
                steps = np.asarray(analyze.progress['Step'], dtype=np.float32)
                values = np.asarray(analyze.progress['EvalCumRewardMean'])
                num_episodes = int(np.median(np.asarray(analyze.progress['EvalNumEpisodes'])))
                steps, values, _ = moving_avg_std(steps, values, window=window//num_episodes)

                # steps = np.array([r['steps'][0] for r in itertools.chain(*analyze.eval_rollouts_itrs)])
                # values = np.array([np.sum(r['rewards']) for r in itertools.chain(*analyze.eval_rollouts_itrs)])
                #
                # steps, values = zip(*sorted(zip(steps, values), key=lambda k: k[0]))
                # steps, values = zip(*[(s, v) for s, v in zip(steps, values) if np.isfinite(v)])
                #
                # steps, values, _ = moving_avg_std(steps, values, window=window)

                data_interp.add_data(steps, values)
            except:
                continue

            if min_step is None:
                min_step = steps[0]
            if max_step is None:
                max_step = steps[-1]
            min_step = max(min_step, steps[0])
            max_step = min(max_step, steps[-1])

        if len(data_interp.xs) == 0:
            return

        steps = np.r_[min_step:max_step:50.][1:-1]
        values_mean, values_std = data_interp.eval(steps)
        # steps -= min_step

    else:
        raise NotImplementedError

    if convert_to_time:
        steps = (0.25 / 3600.) * steps

    if not hasattr(success_cumreward, '__iter__'):
        success_cumreward = [success_cumreward]

    threshold_steps = []
    for i, sc in enumerate(success_cumreward):
        if values_mean.max() >= sc:
            thresh_step = steps[(values_mean >= sc).argmax()]
        else:
            thresh_step = -1

        threshold_steps.append(thresh_step)

    final_step = steps[-1]
    final_value = values_mean[-1]

    return threshold_steps, final_step, final_value

def plot_cumreward_probcoll(ax, analyze_group, color='k', label=None, window=20, success_cumreward=None, ylim=(10, 60)):
    data_interp = DataAverageInterpolation()
    min_step = max_step = None
    for i, analyze in enumerate(analyze_group):
        # steps = np.array(analyze.progress['Step'])
        # values = np.array(analyze.progress['EvalCumRewardMean'])

        steps = np.array(analyze['steps'])
        values = np.array([np.mean([np.sum(r) for r in rewards_itr]) for rewards_itr in analyze['rewards']])

        # steps, values = zip(*sorted(zip(steps, values), key=lambda k: k[0]))
        # steps, values = zip(*[(s, v) for s, v in zip(steps, values) if np.isfinite(v)])
        #
        # def moving_avg_std(idxs, data, window):
        #     avg_idxs, means, stds = [], [], []
        #     for i in range(window, len(data)):
        #         avg_idxs.append(np.mean(idxs[i - window:i]))
        #         means.append(np.mean(data[i - window:i]))
        #         stds.append(np.std(data[i - window:i]))
        #     return avg_idxs, np.asarray(means), np.asarray(stds)
        #
        # steps, values, _ = moving_avg_std(steps, values, window=window)

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

    ax.set_xticks(np.arange(0, max(steps), 1e3), minor=True)
    ax.grid(which='minor', alpha=0.2)
    ax.grid(which='major', alpha=0.5)
    xfmt = ticker.ScalarFormatter()
    xfmt.set_powerlimits((0, 0))
    ax.xaxis.set_major_formatter(xfmt)
    if ylim is not None:
        ax.set_ylim(ylim)

    if success_cumreward is not None:
        if not hasattr(success_cumreward, '__iter__'):
            success_cumreward = [success_cumreward]

        for i, sc in enumerate(success_cumreward):
            if values_mean.max() >= sc:
                thresh_step = steps[(values_mean >= sc).argmax()]
                ax.vlines(thresh_step, *ax.get_ylim(),
                          color=cm.viridis(i / float(len(success_cumreward))), linestyle='--')

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
                if len(pos) < 998:
                    ax.plot([pos[-1, 0]], [pos[-1, 1]], color=color, marker='d')

            ax.set_xlim([-22.5, 22.5])
            ax.set_ylim([-22.5, 22.5])

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

            for pos in positions_i[::len(positions_i) // 10]:
                ax.plot(pos[:, 0], pos[:, 1], color=color, linewidth=1.)
                if len(pos) < 998:
                    ax.plot([pos[-1, 0]], [pos[-1, 1]], color=color, marker='d')

            ax.set_xlim([-22.5, 22.5])
            ax.set_ylim([-22.5, 22.5])

def plot_paths_batch(exp, fname):
    eval_batch_size = exp.params['alg']['batch']['eval_samples_per_batch']
    all_rollouts = np.array(list(itertools.chain(*exp.eval_rollouts_itrs))).reshape(-1, eval_batch_size)

    for i, rollouts in enumerate(all_rollouts):
        divisors = [j for j in range(1, len(rollouts) + 1) if len(rollouts) % j == 0]
        rows = divisors[len(divisors)//2]
        cols = len(rollouts) // rows
        f, axes = plt.subplots(rows, cols, figsize=(2*cols, 2*rows), sharex=True, sharey=True)

        for ax, rollout in zip(axes.ravel(), rollouts):
            pos = np.array([d['pos'][:2] for d in rollout['env_infos']])
            is_collision = (len(pos) < 998) # TODO: hack
            color = 'r' if is_collision else 'b'
            ax.plot(pos[:, 0], pos[:, 1], color=color)
            ax.plot(pos[0, 0], pos[0, 1], color=color, marker='o')

            ax.set_xlim([-22.5, 22.5])
            ax.set_ylim([-22.5, 22.5])

        f.savefig(os.path.join(SAVE_FOLDER, '{0}_{1}_paths.png'.format(fname, i)), bbox_inches='tight', dpi=150)
        plt.close(f)


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

    window = 20

    for ax_cumreward, exp in zip(axes_cumreward.ravel(), all_exps):

        if not hasattr(exp, '__len__'):
            exp = [exp]

        if len(exp) > 0:
            try:
                plot_cumreward(ax_cumreward, exp, window=window, success_cumreward=40.)
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

def plot_1156_1200():
    FILE_NAME = 'rccar_1156_1200'

    le = load_experiments
    base_exp = le(range(1156, 1156 + 3))
    all_exps = [
        base_exp, le(range(1159, 1159 + 3)), [], [],
        base_exp, le(range(1162, 1162 + 3)), le(range(1165, 1165 + 3)), [],
        base_exp, le(range(1168, 1168 + 3)), [], [],
        base_exp, le(range(1171, 1171 + 3)), le(range(1174, 1174 + 3)), le(range(1177, 1177 + 3)),
        base_exp, le(range(1180, 1180 + 3)), le(range(1183, 1183 + 3)), le(range(1186, 1186 + 3)),
        base_exp, le(range(1189, 1189 + 3)), [], [],
        base_exp, le(range(1192, 1192 + 3)), le(range(1195, 1195 + 3)), [],
        base_exp, le(range(1198, 1198 + 3)), [], []
    ]
    title_str_funcs = [
        *[lambda p: '{0}, learn after: {1}'.format(p['exp_name'], p['alg']['learn_after_n_steps'])] * 4,
        *[lambda p: '{0}, train every: {1}'.format(p['exp_name'], p['alg']['train_every_n_steps'])] * 4,
        *[lambda p: '{0}, rp: {1}, coll weight: {2}'.format(p['exp_name'], p['alg']['replay_pool_sampling'], p['policy']['RCcarMACPolicy']['coll_weight_pct'])] * 4,
        *[lambda p: '{0}, target: {1}, clip: {2}'.format(p['exp_name'], p['policy']['use_target'], p['policy']['clip_cost_target_with_dones'])] * 4,
        *[lambda p: '{0}, speed weight: {1}'.format(p['exp_name'], p['policy']['RCcarMACPolicy']['speed_weight'])] * 4,
        *[lambda p: '{0}, increase: {1}'.format(p['exp_name'], p['policy']['RCcarMACPolicy']['probcoll_strictly_increasing'])] * 4,
        *[lambda p: '{0}, lr: {1}'.format(p['exp_name'], p['policy']['lr_schedule']['outside_value'])] * 4,
        *[lambda p: '{0}, batch: {1}'.format(p['exp_name'], p['policy']['MACPolicy']['image_graph']['use_batch_norm'])] * 4,
    ]

    maql_exp = load_experiments(range(994, 994 + 3))
    probcoll_exp = load_probcoll_experiments('/home/gkahn/code/probcoll/experiments/sim_rccar/test/analysis_images\
')

    f_cumreward, axes_cumreward = plt.subplots(8, 4, figsize=(16, 16), sharey=True, sharex=True)

    for ax_cumreward, exp, title_str_func in \
            zip(axes_cumreward.ravel(), all_exps, title_str_funcs):

        if not hasattr(exp, '__len__'):
            exp = [exp]

        if len(exp) > 0:
            try:
                if probcoll_exp is not None:
                    plot_cumreward_probcoll(ax_cumreward, probcoll_exp)
                if maql_exp is not None:
                    plot_cumreward(ax_cumreward, maql_exp, window=8, success_cumreward=40., color='r')
                plot_cumreward(ax_cumreward, exp, window=8, success_cumreward=40.)
                params = exp[0].params
                ax_cumreward.set_title(title_str_func(params), fontdict={'fontsize': 6})
            except:
                pass

    f_cumreward.savefig(os.path.join(SAVE_FOLDER, '{0}_cumreward.png'.format(FILE_NAME)), bbox_inches='tight', dpi\
=300)

    plt.close(f_cumreward)

def plot_1202_1297():
    FILE_NAME = 'rccar_1202_1297'

    all_exps = np.array([load_experiments(range(i, i + 3)) for i in range(1202, 1297, 3)])

    probcoll_exp = load_probcoll_experiments('/home/gkahn/code/probcoll/experiments/sim_rccar/test/analysis_images')
    nstep_exp = load_experiments(range(982, 982+ 3))

    # import IPython; IPython.embed()
    window = 20

    for i, exps in enumerate(np.split(all_exps, 2)):

        f_cumreward, axes_cumreward = plt.subplots(4, 4, figsize=(15, 15), sharey=True, sharex=True)

        for ax_cumreward, exp in zip(axes_cumreward.ravel(), exps):

            if not hasattr(exp, '__len__'):
                exp = [exp]

            if len(exp) > 0:
                try:
                    plot_cumreward(ax_cumreward, exp, window=window, success_cumreward=40.)
                    if probcoll_exp is not None:
                        plot_cumreward_probcoll(ax_cumreward, probcoll_exp)
                    if nstep_exp is not None:
                        plot_cumreward(ax_cumreward, nstep_exp, window=window, success_cumreward=40., color='r')
                    params = exp[0].params
                    for ax in (ax_cumreward,):
                        ax.set_title('{0}, clip: {1}, batch: {2},\nln: {3}, incr: {4}, lr: {5}'.format(
                            params['exp_name'],
                            params['policy']['clip_cost_target_with_dones'],
                            params['policy']['MACPolicy']['image_graph']['use_batch_norm'],
                            params['policy']['MACPolicy']['rnn_graph']['cell_args']['use_layer_norm'],
                            params['policy']['RCcarMACPolicy']['probcoll_strictly_increasing'],
                            params['policy']['lr_schedule']['outside_value']
                        ), fontdict={'fontsize': 6})
                except:
                    pass

        f_cumreward.savefig(os.path.join(SAVE_FOLDER, '{0}_cumreward_{1}.png'.format(FILE_NAME, i)), bbox_inches='tight', dpi=200)
        plt.close(f_cumreward)

def plot_1299_1319():
    FILE_NAME = 'rccar_1299_1319'

    all_exps = [load_experiments(range(1299, 1299 + 3))] + 4*[[]] + \
               [[]] + [load_experiments(range(i, i + 3)) for i in (1302, 1305, 1308, 1311)] + \
               [[]] + [load_experiments(range(i, i + 3)) for i in (1314, 1317)] + [[]] + [[]]

    probcoll_exp = load_probcoll_experiments('/home/gkahn/code/probcoll/experiments/sim_rccar/test/analysis_images')

    f_cumreward, axes_cumreward = plt.subplots(3, 5, figsize=(20, 12), sharey=True, sharex=True)

    window = 20

    for ax_cumreward, exp in zip(axes_cumreward.ravel(), all_exps):

        if not hasattr(exp, '__len__'):
            exp = [exp]

        if len(exp) > 0:
            try:
                plot_cumreward(ax_cumreward, exp, window=window, success_cumreward=40.)
                if probcoll_exp is not None:
                    plot_cumreward_probcoll(ax_cumreward, probcoll_exp)
                params = exp[0].params
                for ax in (ax_cumreward,):
                    ax.set_title('{0}, {1}, N: {2}, H: {3}, speeds: {4}'.format(
                        params['exp_name'],
                        params['policy']['class'],
                        params['policy']['N'],
                        params['policy']['H'],
                        params['alg']['env_eval'].split("'speed_limits':")[-1].split('}')[0]
                    ), fontdict={'fontsize': 6})
            except:
                pass

    f_cumreward.savefig(os.path.join(SAVE_FOLDER, '{0}_cumreward.png'.format(FILE_NAME)), bbox_inches='tight', dpi=150)
    plt.close(f_cumreward)

def plot_comparison_1319():
    FILE_NAME = 'rccar_comparison_1319'

    all_exps = [load_experiments(range(1299, 1299 + 3)), # DQN
                load_experiments(range(1305, 1305 + 3)), # n-step DQN
                load_experiments(range(1317, 1317 + 3)), # MAQL,
                load_experiments(range(1241, 1241 + 3))] # probcoll

    all_labels = ['DQN', 'N-step DQN', 'MAQL', 'MAQL-probcoll']
    all_colors = ['k', 'm', 'r', 'b']

    probcoll_exp = load_probcoll_experiments('/home/gkahn/code/probcoll/experiments/sim_rccar/test/analysis_images')
    probcoll_label = 'probcoll'
    probcoll_color = 'c'

    f_cumreward, ax_cumreward = plt.subplots(1, 1, figsize=(10, 10), sharey=True, sharex=True)

    for exp, color, label in zip(all_exps, all_colors, all_labels):
        plot_cumreward(ax_cumreward, exp, window=20, success_cumreward=40., color=color, label=label)

    # plot_cumreward_probcoll(ax_cumreward, probcoll_exp, color=probcoll_color, label=probcoll_label)

    ax_cumreward.legend(loc='lower right')

    f_cumreward.savefig(os.path.join(SAVE_FOLDER, '{0}_cumreward.png'.format(FILE_NAME)), bbox_inches='tight', dpi=150)
    plt.close(f_cumreward)

def plot_1321_1341():
    FILE_NAME = 'rccar_1321_1341'

    all_exps = [load_experiments(range(i, i + 3)) for i in range(1321, 1338, 3)] + [load_experiments(range(1339, 1339 + 3))]

    f_cumreward, axes_cumreward = plt.subplots(2, 4, figsize=(16, 8), sharey=True, sharex=True)

    window = 20

    for ax_cumreward, exp in zip(axes_cumreward.ravel(), all_exps):

        if not hasattr(exp, '__len__'):
            exp = [exp]

        if len(exp) > 0:
            try:
                plot_cumreward(ax_cumreward, exp, window=window, success_cumreward=40., ylim=None)
                params = exp[0].params
                for ax in (ax_cumreward,):
                    ax.set_title('{0}, {1}, N: {2}, H: {3}, speeds: {4}'.format(
                        params['exp_name'],
                        params['policy']['class'],
                        params['policy']['N'],
                        params['policy']['H'],
                        params['alg']['env_eval'].split("'speed_limits':")[-1].split('}')[0]
                    ), fontdict={'fontsize': 6})
            except:
                pass

    f_cumreward.savefig(os.path.join(SAVE_FOLDER, '{0}_cumreward.png'.format(FILE_NAME)), bbox_inches='tight', dpi=150)
    plt.close(f_cumreward)

def plot_1339_paths():
    analyze = AnalyzeRNNCritic(os.path.join(EXP_FOLDER, 'rccar1339'),
                               clear_obs=False,
                               create_new_envs=True,
                               load_train_rollouts=True,
                               load_eval_rollouts=True)

    rollouts = analyze.eval_policy(5, gpu_device=0, gpu_frac=0.5)

    import IPython; IPython.embed()

def plot_1343_1366():
    FILE_NAME = 'rccar_1343_1366'

    all_exps = [load_experiments(range(i, i + 3)) for i in range(1343, 1366, 3)] + \
               [[load_probcoll_experiments('/home/gkahn/code/rllab/data/s3/sim-rccar/', i) for i in range(657, 657 + 3)]]

    f_cumreward, axes_cumreward = plt.subplots(3, 3, figsize=(16, 8), sharey=True, sharex=True)

    window = 20
    probcoll_window = 4
    ylim = (0, 2100)
    success_cumreward = [500, 1000, 1500, 1750]

    for ax_cumreward, exp in zip(axes_cumreward.ravel(), all_exps):

        if not hasattr(exp, '__len__'):
            exp = [exp]

        if len(exp) > 0:
            try:
                if type(exp[0]) is dict:
                    plot_cumreward_probcoll(ax_cumreward, exp, window=probcoll_window, success_cumreward=success_cumreward, ylim=ylim)
                    for ax in (ax_cumreward,):
                        ax.set_title('probcoll {0}'.format(exp[0]['exp_num']),
                                     fontdict={'fontsize': 6})
                else:
                    plot_cumreward(ax_cumreward, exp, window=window, success_cumreward=success_cumreward, ylim=ylim)
                    params = exp[0].params
                    for ax in (ax_cumreward,):
                        ax.set_title('{0}, {1}, N: {2}, H: {3}, speeds: {4}'.format(
                            params['exp_name'],
                            params['policy']['class'],
                            params['policy']['N'],
                            params['policy']['H'],
                            params['alg']['env_eval'].split("'speed_limits':")[-1].split('}')[0]
                        ), fontdict={'fontsize': 6})
            except:
                pass

    f_cumreward.savefig(os.path.join(SAVE_FOLDER, '{0}_cumreward.png'.format(FILE_NAME)), bbox_inches='tight', dpi=150)
    plt.close(f_cumreward)

def plot_1368_1382():
    FILE_NAME = 'rccar_1368_1382'

    all_exps = [load_experiments(range(i, i + 3)) for i in range(1368, 1382, 3)] + \
               [[load_probcoll_experiments('/home/gkahn/code/rllab/data/s3/sim-rccar/', i) for i in range(663, 663 + 3)]]

    f_cumreward, axes_cumreward = plt.subplots(3, 3, figsize=(16, 8), sharey=True, sharex=True)

    window = 20
    probcoll_window = 4
    ylim = (0, 2100)
    success_cumreward = [500, 1000, 1500, 1750]

    for ax_cumreward, exp in zip(axes_cumreward.ravel(), all_exps):

        if not hasattr(exp, '__len__'):
            exp = [exp]

        if len(exp) > 0:
            try:
                if type(exp[0]) is dict:
                    plot_cumreward_probcoll(ax_cumreward, exp, window=probcoll_window, success_cumreward=success_cumreward, ylim=ylim)
                    for ax in (ax_cumreward,):
                        ax.set_title('probcoll {0}'.format(exp[0]['exp_num']),
                                     fontdict={'fontsize': 6})
                else:
                    plot_cumreward(ax_cumreward, exp, window=window, success_cumreward=success_cumreward, ylim=ylim)
                    params = exp[0].params
                    for ax in (ax_cumreward,):
                        ax.set_title('{0}, {1}, N: {2}, H: {3}, speeds: {4}'.format(
                            params['exp_name'],
                            params['policy']['class'],
                            params['policy']['N'],
                            params['policy']['H'],
                            params['alg']['env_eval'].split("'speed_limits':")[-1].split('}')[0]
                        ), fontdict={'fontsize': 6})
            except:
                pass

    f_cumreward.savefig(os.path.join(SAVE_FOLDER, '{0}_cumreward.png'.format(FILE_NAME)), bbox_inches='tight', dpi=150)
    plt.close(f_cumreward)

def plot_1384_1431():
    FILE_NAME = 'rccar_1384_1431'

    all_exps = [load_experiments(range(i, i + 3)) for i in range(1384, 1431, 3)]

    f_cumreward, axes_cumreward = plt.subplots(4, 4, figsize=(16, 16), sharey=True, sharex=True)

    window = 20
    ylim = (0, 2100)
    success_cumreward = [500, 1000, 1500, 1750]

    for ax_cumreward, exp in zip(axes_cumreward.ravel(), all_exps):

        if not hasattr(exp, '__len__'):
            exp = [exp]

        if len(exp) > 0:
            try:
                plot_cumreward(ax_cumreward, exp, window=window, success_cumreward=success_cumreward, ylim=ylim)
                params = exp[0].params
                for ax in (ax_cumreward,):
                    ax.set_title('{0}, train: {1}, update: {2:.1e}, expl: {3:.1e}, rp: {4}'.format(
                        params['exp_name'],
                        params['alg']['train_every_n_steps'],
                        params['alg']['update_target_every_n_steps'],
                        params['alg']['exploration_strategies']['GaussianStrategy']['endpoints'][-1][0],
                        params['alg']['replay_pool_sampling']
                    ), fontdict={'fontsize': 6})
            except:
                pass

    f_cumreward.savefig(os.path.join(SAVE_FOLDER, '{0}_cumreward.png'.format(FILE_NAME)), bbox_inches='tight', dpi=150)
    plt.close(f_cumreward)

def plot_1433_1492():
    FILE_NAME = 'rccar_1433_1492'

    all_exps = [load_experiments(range(i, i + 3)) for i in range(1433, 1448, 3)] + \
               [[load_probcoll_experiments('/home/gkahn/code/rllab/data/s3/sim-rccar/', i) for i in range(667, 667 + 3)]] + \
               [load_experiments(range(i, i + 3)) for i in range(1448, 1463, 3)] + \
               [[load_probcoll_experiments('/home/gkahn/code/rllab/data/s3/sim-rccar/', i) for i in
                 range(670, 670 + 3)]] + \
               [load_experiments(range(i, i + 3)) for i in range(1463, 1478, 3)] + \
               [[load_probcoll_experiments('/home/gkahn/code/rllab/data/s3/sim-rccar/', i) for i in
                 range(673, 673 + 3)]] + \
               [load_experiments(range(i, i + 3)) for i in range(1478, 1493, 3)] + \
               [[load_probcoll_experiments('/home/gkahn/code/rllab/data/s3/sim-rccar/', i) for i in
                 range(676, 676 + 3)]]

    import IPython; IPython.embed()

    f_cumreward, axes_cumreward = plt.subplots(4, 6, figsize=(18, 12), sharey=True, sharex=False)

    window = 20
    probcoll_window = 4
    ylim = (0, 2100)
    success_cumreward = [500, 1000, 1500, 1750]

    for ax_cumreward, exp in zip(axes_cumreward.ravel(), all_exps):

        if not hasattr(exp, '__len__'):
            exp = [exp]

        if len(exp) > 0:
            try:
                if type(exp[0]) is dict:
                    plot_cumreward_probcoll(ax_cumreward, exp, window=probcoll_window, success_cumreward=success_cumreward, ylim=ylim)
                    for ax in (ax_cumreward,):
                        ax.set_title('probcoll {0}'.format(exp[0]['exp_num']),
                                     fontdict={'fontsize': 6})
                else:
                    plot_cumreward(ax_cumreward, exp, window=window, success_cumreward=success_cumreward, ylim=ylim)
                    params = exp[0].params
                    for ax in (ax_cumreward,):
                        ax.set_title('{0}, {1}, {2}, N: {3}, H: {4}'.format(
                            params['exp_name'],
                            params['alg']['env'].split('(params=')[0].split('"')[-1],
                            params['policy']['class'],
                            params['policy']['N'],
                            params['policy']['H'],
                        ), fontdict={'fontsize': 6})
            except:
                pass

    for i, xmax in enumerate([1e5, 2e5, 2e5, 4e5]):
        for ax in axes_cumreward[i, :]:
            ax.set_xlim((0, xmax))


    f_cumreward.savefig(os.path.join(SAVE_FOLDER, '{0}_cumreward.png'.format(FILE_NAME)), bbox_inches='tight', dpi=150)
    plt.close(f_cumreward)

def plot_1494_1553():
    FILE_NAME = 'rccar_1494_1553'

    all_exps = [load_experiments(range(i, i + 3)) for i in range(1494, 1509, 3)] + \
               [[load_probcoll_experiments('/home/gkahn/code/rllab/data/s3/sim-rccar/', i) for i in range(680, 680+ 3)]] + \
               [load_experiments(range(i, i + 3)) for i in range(1509, 1524, 3)] + \
               [[load_probcoll_experiments('/home/gkahn/code/rllab/data/s3/sim-rccar/', i) for i in range(683, 683+ 3)]] + \
               [load_experiments(range(i, i + 3)) for i in range(1524, 1539, 3)] + \
               [[load_probcoll_experiments('/home/gkahn/code/rllab/data/s3/sim-rccar/', i) for i in range(686, 686+ 3)]] + \
               [load_experiments(range(i, i + 3)) for i in range(1539, 1554, 3)] + \
               [[load_probcoll_experiments('/home/gkahn/code/rllab/data/s3/sim-rccar/', i) for i in range(689, 689+ 3)]]

    # import IPython; IPython.embed()

    f_cumreward, axes_cumreward = plt.subplots(4, 6, figsize=(18, 12), sharey=True, sharex=False)

    window = 20
    probcoll_window = 4
    ylim = (0, 2100)
    success_cumreward = [500, 1000, 1500, 1750]

    for ax_cumreward, exp in zip(axes_cumreward.ravel(), all_exps):

        if not hasattr(exp, '__len__'):
            exp = [exp]

        if len(exp) > 0:
            try:
                if type(exp[0]) is dict:
                    plot_cumreward_probcoll(ax_cumreward, exp, window=probcoll_window, success_cumreward=success_cumreward, ylim=ylim)
                    for ax in (ax_cumreward,):
                        ax.set_title('probcoll {0}'.format(exp[0]['exp_num']),
                                     fontdict={'fontsize': 6})
                else:
                    plot_cumreward(ax_cumreward, exp, window=window, success_cumreward=success_cumreward, ylim=ylim)
                    params = exp[0].params
                    for ax in (ax_cumreward,):
                        ax.set_title('{0}, {1}, {2}, N: {3}, H: {4}'.format(
                            params['exp_name'],
                            params['alg']['env'].split('(params=')[0].split('"')[-1],
                            params['policy']['class'],
                            params['policy']['N'],
                            params['policy']['H'],
                        ), fontdict={'fontsize': 6})
            except:
                pass

    for i, xmax in enumerate([1e5, 2e5, 2e5, 4e5]):
        for ax in axes_cumreward[i, :]:
            ax.set_xlim((0, xmax))


    f_cumreward.savefig(os.path.join(SAVE_FOLDER, '{0}_cumreward.png'.format(FILE_NAME)), bbox_inches='tight', dpi=150)
    plt.close(f_cumreward)

def plot_1555_1650():
    FILE_NAME = 'rccar_1555_1650'

    all_exps = np.array([load_experiments(range(i, i + 3)) for i in range(1555, 1650, 3)])

    import IPython; IPython.embed()

    window = 20
    ylim = (0, 2100)
    success_cumreward = [500, 1000, 1500, 1750]

    for i, exps in enumerate(np.split(all_exps, 2)):

        f_cumreward, axes_cumreward = plt.subplots(4, 4, figsize=(15, 15), sharey=True, sharex=True)

        for ax_cumreward, exp in zip(axes_cumreward.ravel(), exps):

            if not hasattr(exp, '__len__'):
                exp = [exp]

            if len(exp) > 0:
                plot_cumreward(ax_cumreward, exp, window=window, success_cumreward=success_cumreward, ylim=ylim)
                params = exp[0].params
                for ax in (ax_cumreward,):
                    ax.set_title('{0}, {1}, reg: {2}, train: {3}, lr: {4}'.format(
                        params['exp_name'],
                        params['policy']['get_action_test']['type'],
                        params['policy']['weight_decay'],
                        params['alg']['train_every_n_steps'],
                        params['policy']['lr_schedule']['outside_value']

                    ), fontdict={'fontsize': 6})

        f_cumreward.savefig(os.path.join(SAVE_FOLDER, '{0}_cumreward_{1}.png'.format(FILE_NAME, i)), bbox_inches='tight', dpi=200)
        plt.close(f_cumreward)

def plot_1652_1795():
    FILE_NAME = 'rccar_1652_1795'

    all_exps = np.array([load_experiments(range(i, i + 3)) for i in range(1652, 1795, 3)])

    import IPython; IPython.embed()

    window = 20
    ylim = (0, 2100)
    success_cumreward = [500, 1000, 1500, 1750]

    for i, exps in enumerate(np.split(all_exps, 4)):

        f_cumreward, axes_cumreward = plt.subplots(4, 3, figsize=(9, 12), sharey=True, sharex=True)

        for ax_cumreward, exp in zip(axes_cumreward.ravel(), exps):

            if not hasattr(exp, '__len__'):
                exp = [exp]

            if len(exp) > 0:
                plot_cumreward(ax_cumreward, exp, window=window, success_cumreward=success_cumreward, ylim=ylim)
                params = exp[0].params
                for ax in (ax_cumreward,):
                    ax.set_title('{0}, {1}, train every: {2},\ncell: {3}, norm: {4}, reg: {5:.1e}'.format(
                        params['exp_name'],
                        params['alg']['type'],
                        params['alg']['interleave']['train_every_n_steps'] if params['alg']['type'] == 'interleave' else '',
                        params['policy']['MACPolicy']['rnn_graph']['cell_type'],
                        params['policy']['MACPolicy']['image_graph']['normalizer'],
                        params['policy']['weight_decay']
                    ), fontdict={'fontsize': 6})

        f_cumreward.savefig(os.path.join(SAVE_FOLDER, '{0}_cumreward_{1}.png'.format(FILE_NAME, i)), bbox_inches='tight',
                            dpi=200)
        plt.close(f_cumreward)

def plot_1797_1904():
    FILE_NAME = 'rccar_1797_1904'

    all_exps = np.array([load_experiments(range(i, i + 3)) for i in range(1797, 1904, 3)])

    # import IPython; IPython.embed()

    window = 20
    ylim = (0, 2100)
    success_cumreward = [500, 1000, 1500, 1750]

    for i, exps in enumerate(np.split(all_exps, 3)):

        f_cumreward, axes_cumreward = plt.subplots(3, 4, figsize=(12, 9), sharey=True, sharex=True)

        for ax_cumreward, exp in zip(axes_cumreward.ravel(), exps):

            if not hasattr(exp, '__len__'):
                exp = [exp]

            if len(exp) > 0:
                # try:
                plot_cumreward(ax_cumreward, exp, window=window, success_cumreward=success_cumreward, ylim=ylim)
                # except:
                #     pass
                params = exp[0].params
                for ax in (ax_cumreward,):
                    ax.set_title('{0}, {1}, total batches: {2}, steps per: {3},\ntrain steps per: {4}, reg: {5:.1e}'.format(
                        params['exp_name'],
                        params['alg']['type'],
                        params['alg']['batch']['total_batches'],
                        params['alg']['batch']['steps_per_batch'],
                        params['alg']['batch']['train_steps_per_batch'],
                        params['policy']['weight_decay']
                    ), fontdict={'fontsize': 6})

        f_cumreward.savefig(os.path.join(SAVE_FOLDER, '{0}_cumreward_{1}.png'.format(FILE_NAME, i)), bbox_inches='tight',
                            dpi=200)
        plt.close(f_cumreward)

def plot_compare_1494_1905():
    FILE_NAME = 'rccar_1905_1494_compare'

    all_exps = [load_experiments(range(1494, 1494 + 3)), load_experiments(range(1496, 1496 + 3)),
                load_experiments(range(1905, 1905 + 3)), load_experiments(range(1908, 1908 + 3))]

    # import IPython; IPython.embed()

    f_cumreward, axes_cumreward = plt.subplots(2, 2, figsize=(10, 5), sharey=True, sharex=True)

    window = 20
    ylim = (0, 2100)
    success_cumreward = [500, 1000, 1500, 1750]

    for ax_cumreward, exp in zip(axes_cumreward.ravel(), all_exps):

        if not hasattr(exp, '__len__'):
            exp = [exp]

        if len(exp) > 0:
            plot_cumreward(ax_cumreward, exp, window=window, success_cumreward=success_cumreward, ylim=ylim)
            params = exp[0].params
            for ax in (ax_cumreward,):
                ax.set_title('{0}'.format(
                    params['exp_name'],
                ), fontdict={'fontsize': 8})

    f_cumreward.savefig(os.path.join(SAVE_FOLDER, '{0}_cumreward.png'.format(FILE_NAME)), bbox_inches='tight', dpi=150)
    plt.close(f_cumreward)

def plot_1912_2055():
    FILE_NAME = 'rccar_1912_2055'

    all_exps = np.array([load_experiments(range(i, i + 3)) for i in range(1912, 2055, 3)])

    import IPython; IPython.embed()

    window = 20
    ylim = (0, 2100)
    success_cumreward = [500, 1000, 1500, 1750]

    for i, exps in enumerate(np.split(all_exps, 2)):

        f_cumreward, axes_cumreward = plt.subplots(6, 4, figsize=(8, 12), sharey=True, sharex=True)

        for ax_cumreward, exp in zip(axes_cumreward.ravel(), exps):

            if not hasattr(exp, '__len__'):
                exp = [exp]

            if len(exp) > 0:
                try:
                    plot_cumreward(ax_cumreward, exp, window=window, success_cumreward=success_cumreward, ylim=ylim)
                except:
                    pass
                params = exp[0].params
                for ax in (ax_cumreward,):
                    ax.set_title('{0}, {1}, total batches: {2}, steps per: {3},\ntrain steps per: {4}, norm: {5}, reg: {6:.1e}'.format(
                        params['exp_name'],
                        params['alg']['type'],
                        params['alg']['batch']['total_batches'],
                        params['alg']['batch']['steps_per_batch'],
                        params['alg']['batch']['train_steps_per_batch'],
                        params['policy']['MACPolicy']['image_graph']['normalizer'],
                        params['policy']['weight_decay']
                    ), fontdict={'fontsize': 4})

        f_cumreward.savefig(os.path.join(SAVE_FOLDER, '{0}_cumreward_{1}.png'.format(FILE_NAME, i)), bbox_inches='tight',
                            dpi=200)
        plt.close(f_cumreward)

def plot_2057_2128():
    FILE_NAME = 'rccar_2057_2128'

    all_exps = np.array([load_experiments(range(i, i + 3)) for i in range(2057, 2128, 3)])

    # import IPython; IPython.embed()

    window = 20
    ylim = (0, 2100)
    success_cumreward = [500, 1000, 1500, 1750]

    for i, exps in enumerate(np.split(all_exps, 3)):

        f_cumreward, axes_cumreward = plt.subplots(2, 4, figsize=(12, 6), sharey=True, sharex=True)

        for ax_cumreward, exp in zip(axes_cumreward.ravel(), exps):

            if not hasattr(exp, '__len__'):
                exp = [exp]

            if len(exp) > 0:
                try:
                    plot_cumreward(ax_cumreward, exp, window=window, success_cumreward=success_cumreward, ylim=ylim)
                except:
                    pass
                params = exp[0].params
                for ax in (ax_cumreward,):
                    ax.set_title('{0}, {1}, reset: {2},\nnorm: {3}, reg: {4:.1e}'.format(
                        params['exp_name'],
                        params['alg']['type'],
                        params['alg']['batch']['reset_every_n_batches'],
                        params['policy']['MACPolicy']['image_graph']['normalizer'],
                        params['policy']['weight_decay']
                    ), fontdict={'fontsize': 7})

        f_cumreward.savefig(os.path.join(SAVE_FOLDER, '{0}_cumreward_{1}.png'.format(FILE_NAME, i)), bbox_inches='tight',
                            dpi=200)
        plt.close(f_cumreward)

def plot_2130_2201():
    FILE_NAME = 'rccar_2130_2201'

    all_exps = np.array([load_experiments(range(i, i + 3)) for i in range(2130, 2201, 3)])

    # import IPython; IPython.embed()

    window = 20
    ylim = (0, 2100)
    success_cumreward = [500, 1000, 1500, 1750]

    for i, exps in enumerate(np.split(all_exps, 2)):

        f_cumreward, axes_cumreward = plt.subplots(4, 3, figsize=(9, 12), sharey=True, sharex=True)

        for ax_cumreward, exp in zip(axes_cumreward.ravel(), exps):

            if not hasattr(exp, '__len__'):
                exp = [exp]

            if len(exp) > 0:
                try:
                    plot_cumreward(ax_cumreward, exp, window=window, success_cumreward=success_cumreward, ylim=ylim)
                except:
                    pass
                params = exp[0].params
                for ax in (ax_cumreward,):
                    ax.set_title('{0}, {1}, reset: {2},\ndropout: {3}, reg: {4:.1e}'.format(
                        params['exp_name'],
                        params['alg']['type'],
                        params['alg']['batch']['reset_every_n_batches'],
                        params['policy']['MACPolicy']['observation_graph']['dropout'],
                        params['policy']['weight_decay']
                    ), fontdict={'fontsize': 7})

        f_cumreward.savefig(os.path.join(SAVE_FOLDER, '{0}_cumreward_{1}.png'.format(FILE_NAME, i)), bbox_inches='tight',
                            dpi=200)
        plt.close(f_cumreward)

def plot_2203_2346():
    FILE_NAME = 'rccar_2203_2346'

    all_exps = np.array([load_experiments(range(i, i + 3)) for i in range(2203, 2346, 3)])

    # import IPython; IPython.embed()

    window = 20
    ylim = (0, 2100)
    success_cumreward = [500, 1000, 1500, 1750]

    for i, exps in enumerate(np.split(all_exps, 4)):

        f_cumreward, axes_cumreward = plt.subplots(3, 4, figsize=(12, 9), sharey=True, sharex=True)

        for ax_cumreward, exp in zip(axes_cumreward.ravel(), exps):

            if not hasattr(exp, '__len__'):
                exp = [exp]

            if len(exp) > 0:
                try:
                    plot_cumreward(ax_cumreward, exp, window=window, success_cumreward=success_cumreward, ylim=ylim)
                except:
                    pass
                params = exp[0].params
                for ax in (ax_cumreward,):
                    ax.set_title('{0}, {1}, H: {2}, {3}, frac: {4},\nclip: {5}, reg: {6:.1e}'.format(
                        params['exp_name'],
                        params['alg']['type'],
                        params['policy']['H'],
                        params['policy']['get_action_test']['type'],
                        params['alg']['replay_pool_params']['terminal']['frac'],
                        params['policy']['clip_cost_target_with_dones'],
                        params['policy']['weight_decay']
                    ), fontdict={'fontsize': 6})

        f_cumreward.savefig(os.path.join(SAVE_FOLDER, '{0}_cumreward_{1}.png'.format(FILE_NAME, i)), bbox_inches='tight',
                            dpi=200)
        plt.close(f_cumreward)

def plot_compare():
    FILE_NAME = 'rccar_compare'

    # exps = [load_experiments(range(1497, 1497 + 3)),
    #         load_experiments(range(2108, 2108 + 3), create_new_envs=True)]

    # import IPython; IPython.embed()

    ### cumreward
    # f_cumreward, ax_cumreward = plt.subplots(1, 2, figsize=(20, 10), sharey=True, sharex=True)
    #
    # window = 20
    # ylim = (0, 2100)
    # success_cumreward = [500, 1000, 1500, 1750]
    #
    # plot_cumreward(ax_cumreward[0], exps[0], window=window, success_cumreward=success_cumreward, color='k', ylim=ylim)
    # plot_cumreward(ax_cumreward[1], exps[1], window=window, success_cumreward=success_cumreward, color='b', ylim=ylim)
    #
    # f_cumreward.savefig(os.path.join(SAVE_FOLDER, '{0}_cumreward.png'.format(FILE_NAME)), bbox_inches='tight', dpi=150)
    # plt.close(f_cumreward)

    ### paths
    # paths_cols = 6
    # f_paths, axes_paths = plt.subplots(len(exps), paths_cols, figsize=(2*paths_cols, 2*len(exps)), sharey=True, sharex=True)
    # if len(axes_paths.shape) == 1:
    #     axes_paths = np.array([axes_paths])
    #
    # for axes_row, exp in zip(axes_paths, exps):
    #     plot_paths(axes_row, exp)
    #
    # f_paths.savefig(os.path.join(SAVE_FOLDER, '{0}_paths.png'.format(FILE_NAME)), bbox_inches='tight', dpi=150)
    # plt.close(f_paths)

    exp = load_experiments(range(2108, 2108 + 1), create_new_envs=True)[0] # exps[1][0]
    # plot_paths_batch(exp, 'rccar_compare_batch')

    eval_batch_size = exp.params['alg']['batch']['eval_samples_per_batch']
    all_rollouts = np.array(list(itertools.chain(*exp.eval_rollouts_itrs))).reshape(-1, eval_batch_size)

    sess, graph = MACPolicy.create_session_and_graph(gpu_device=0, gpu_frac=0.9)
    with graph.as_default(), sess.as_default():
        policy = exp._load_itr_policy(3)

        rollout = all_rollouts[19, 0]
        env_infos = rollout['env_infos']
        num_obs = 4
        states = [(ei['pos'] + [0., 0, 0], ei['hpr']) for ei in env_infos[-policy.N-num_obs:-policy.N]]
        env = exp.env.wrapped_env.wrapped_env
        observations = [env.reset(pos=pos, hpr=hpr) for pos, hpr in states]
        flat_obs = np.array(observations).reshape(num_obs, -1)

        K = 4096
        actions = np.array([policy._env_spec.action_space.sample() for _ in range(K * (policy.N + 1))]).reshape(K, policy.N + 1, -1)

        values, _ = policy.get_values([flat_obs] * K, actions)
        action_positions = []

        for action_sequence in actions:
            positions = [states[-1][0]]
            env.reset(pos=states[-1][0], hpr=states[-1][1])
            for a in action_sequence[:-1]:
                _, _, done, env_info = env.step(a)
                positions.append(env_info['pos'])
                if done:
                    break
            action_positions.append(positions)

        action_positions, values = zip(*sorted([(p, v) for p, v in zip(action_positions, values)], key=lambda x: x[1].mean()))
        # indices = np.linspace(0, K - 1, 100, dtype=np.int32)
        indices = np.linspace(K - 25 - 1, K - 1, 25, dtype=np.int32)

        import IPython; IPython.embed()

        f, ax = plt.subplots(1, 1, figsize=(5, 5))
        for i in indices:
            positions = np.array(action_positions[i])
            value = values[i]
            color = cm.Greys(-0.8 * value.mean() + 0.2)

            ax.plot(positions[:, 0], positions[:, 1], color=color)
            if len(positions) <= policy.N:
                ax.plot(positions[-1, 0], positions[-1, 1], color='r', marker='x')

            # marker = 'x' if len(positions) <= policy.N else 'o'
            # ax.plot(positions[-1, 0], positions[-1, 1], color=color, marker=marker, markersize=1.5)

        best_index = np.array(values).mean(axis=1).argmax()
        positions = np.array(action_positions[best_index])
        ax.plot(positions[:, 0], positions[:, 1], color='g')

        f.savefig(os.path.join(SAVE_FOLDER, '{0}_cost.png'.format(FILE_NAME)), bbox_inches='tight', dpi=200)
        plt.close(f)

        f, ax = plt.subplots(1, 1, figsize=(5, 5))
        ax.imshow(observations[-1][:, :, 0], cmap='Greys_r')
        f.savefig(os.path.join(SAVE_FOLDER, '{0}_cost_image.png'.format(FILE_NAME)), bbox_inches='tight', dpi=200)
        plt.close(f)

def plot_2348_2443():
    FILE_NAME = 'rccar_2348_2443'

    all_exps = np.array([load_experiments(range(i, i + 3)) for i in range(2348, 2443, 3)])

    import IPython; IPython.embed()

    window = 4
    ylim = (0, 2100)
    success_cumreward = [500, 1000, 1500, 1750]

    for i, exps in enumerate(np.split(all_exps, 2)):

        f_cumreward, axes_cumreward = plt.subplots(4, 4, figsize=(12, 12), sharey=True, sharex=True)

        for ax_cumreward, exp in zip(axes_cumreward.ravel(), exps):

            if not hasattr(exp, '__len__'):
                exp = [exp]

            if len(exp) > 0:
                try:
                    plot_cumreward(ax_cumreward, exp, window=window, success_cumreward=success_cumreward, ylim=ylim)
                except:
                    pass
                params = exp[0].params
                for ax in (ax_cumreward,):
                    ax.set_title('{0}, {1}, H: {2},\n{3}, norm: {4}, reg: {5:.1e}'.format(
                        params['exp_name'],
                        params['alg']['type'],
                        params['policy']['H'],
                        params['policy']['get_action_test']['type'],
                        params['policy']['MACPolicy']['observation_graph']['normalizer'],
                        params['policy']['weight_decay']
                    ), fontdict={'fontsize': 6})

        f_cumreward.savefig(os.path.join(SAVE_FOLDER, '{0}_cumreward_{1}.png'.format(FILE_NAME, i)),
                            bbox_inches='tight',
                            dpi=200)
        plt.close(f_cumreward)

def plot_2445_2516():
    FILE_NAME = 'rccar_2445_2516'

    all_exps = [load_experiments(range(i, i + 3), load_eval_rollouts=False) for i in range(2445, 2516, 3)]

    # import IPython; IPython.embed()

    f_cumreward, axes_cumreward = plt.subplots(4, 6, figsize=(18, 12), sharey=True, sharex=False)

    window = 16
    ylim = (0, 2100)
    success_cumreward = [500, 1000, 1500, 1750]

    for ax_cumreward, exp in zip(axes_cumreward.ravel(), all_exps):

        if not hasattr(exp, '__len__'):
            exp = [exp]

        if len(exp) > 0:
            try:
                plot_cumreward(ax_cumreward, exp, window=window, success_cumreward=success_cumreward, ylim=ylim)
            except:
                pass
            params = exp[0].params
            for ax in (ax_cumreward,):
                ax.set_title('{0}, {1}, {2}, N: {3}, H: {4}'.format(
                    params['exp_name'],
                    params['alg']['env'].split('(params=')[0].split('"')[-1],
                    params['policy']['class'],
                    params['policy']['N'],
                    params['policy']['H'],
                ), fontdict={'fontsize': 6})

    for i, xmax in enumerate([2e5, 4e5, 4e5, 8e5]):
        for ax in axes_cumreward[i, :]:
            ax.set_xlim((0, xmax))

    f_cumreward.savefig(os.path.join(SAVE_FOLDER, '{0}_cumreward.png'.format(FILE_NAME)), bbox_inches='tight', dpi=150)
    plt.close(f_cumreward)

def plot_2518_2577_and_2796_2819():
    FILE_NAME = 'rccar_2518_2577_and_2769_2819'

    all_exps = [load_experiments(range(i, i + 3), load_eval_rollouts=False) for i in range(2518, 2577, 3)]
    all_exps = all_exps[:4] + \
               [load_experiments(range(i, i + 3), load_eval_rollouts=False) for i in range(2796, 2819, 3)] + \
               all_exps[4:]

    # import IPython; IPython.embed()

    f_cumreward, axes_cumreward = plt.subplots(7, 4, figsize=(12, 21), sharey=True, sharex=True)

    window = 16
    ylim = (0, 2100)
    success_cumreward = [500, 1000, 1500, 1750]

    for ax_cumreward, exp in zip(axes_cumreward.ravel(), all_exps):

        if len(exp) > 0:
            # try:
            plot_cumreward(ax_cumreward, exp, window=window, success_cumreward=success_cumreward, ylim=ylim)
            # except:
            #     pass
            params = exp[0].params
            for ax in (ax_cumreward,):
                title = '{0}, {1}, {2}, H: {3},\ntarg: {4}, classif: {5}, incr: {6}, rp: {7}, clip: {8}'.format(
                    params['exp_name'],
                    params['alg']['env'].split('(params=')[0].split('"')[-1],
                    params['policy']['class'],
                    params['policy']['H'],
                    params['policy']['use_target'],
                    params['policy']['RCcarMACPolicy']['is_classification'] if params['policy']['class'] == 'RCcarMACPolicy' else '',
                    params['policy']['RCcarMACPolicy']['probcoll_strictly_increasing'] if params['policy']['class'] == 'RCcarMACPolicy' else '',
                    params['alg']['replay_pool_sampling'],
                    params['policy']['clip_cost_target_with_dones'],
                )

                ax.set_title(title, fontdict={'fontsize': 5})

    f_cumreward.savefig(os.path.join(SAVE_FOLDER, '{0}_cumreward.png'.format(FILE_NAME)), bbox_inches='tight', dpi=150)
    plt.close(f_cumreward)

def plot_2578_2637_and_2820_2843():
    FILE_NAME = 'rccar_2578_2637_and_2820_2843'

    all_exps = [load_experiments(range(i, i + 3), load_eval_rollouts=False) for i in range(2578, 2637, 3)]
    all_exps = all_exps[:4] + \
               [load_experiments(range(i, i + 3), load_eval_rollouts=False) for i in range(2820, 2843, 3)] + \
               all_exps[4:]

    # import IPython; IPython.embed()

    f_cumreward, axes_cumreward = plt.subplots(7, 4, figsize=(12, 21), sharey=True, sharex=True)

    window = 16
    ylim = (0, 2100)
    success_cumreward = [500, 1000, 1500, 1750]

    for ax_cumreward, exp in zip(axes_cumreward.ravel(), all_exps):

        if len(exp) > 0:
            # try:
            plot_cumreward(ax_cumreward, exp, window=window, success_cumreward=success_cumreward, ylim=ylim)
            # except:
            #     pass
            params = exp[0].params
            for ax in (ax_cumreward,):
                title = '{0}, {1}, {2}, H: {3},\ntarg: {4}, classif: {5}, incr: {6}, rp: {7}, clip: {8}'.format(
                    params['exp_name'],
                    params['alg']['env'].split('(params=')[0].split('"')[-1],
                    params['policy']['class'],
                    params['policy']['H'],
                    params['policy']['use_target'],
                    params['policy']['RCcarMACPolicy']['is_classification'] if params['policy']['class'] == 'RCcarMACPolicy' else '',
                    params['policy']['RCcarMACPolicy']['probcoll_strictly_increasing'] if params['policy']['class'] == 'RCcarMACPolicy' else '',
                    params['alg']['replay_pool_sampling'],
                    params['policy']['clip_cost_target_with_dones'],
                )

                ax.set_title(title, fontdict={'fontsize': 5})

    f_cumreward.savefig(os.path.join(SAVE_FOLDER, '{0}_cumreward.png'.format(FILE_NAME)), bbox_inches='tight', dpi=150)
    plt.close(f_cumreward)

def plot_2638_2697_and_2844_2867():
    FILE_NAME = 'rccar_2638_2697_and_2844_2867'

    all_exps = [load_experiments(range(i, i + 3), load_eval_rollouts=False) for i in range(2638, 2697, 3)]
    all_exps = all_exps[:4] + \
               [load_experiments(range(i, i + 3), load_eval_rollouts=False) for i in range(2844, 2867, 3)] + \
               all_exps[4:]

    # import IPython; IPython.embed()

    f_cumreward, axes_cumreward = plt.subplots(7, 4, figsize=(12, 21), sharey=True, sharex=True)

    window = 16
    ylim = (0, 2100)
    success_cumreward = [500, 1000, 1500, 1750]

    for ax_cumreward, exp in zip(axes_cumreward.ravel(), all_exps):

        if len(exp) > 0:
            # try:
            plot_cumreward(ax_cumreward, exp, window=window, success_cumreward=success_cumreward, ylim=ylim)
            # except:
            #     pass
            params = exp[0].params
            for ax in (ax_cumreward,):
                title = '{0}, {1}, {2}, H: {3},\ntarg: {4}, classif: {5}, incr: {6}, rp: {7}, clip: {8}'.format(
                    params['exp_name'],
                    params['alg']['env'].split('(params=')[0].split('"')[-1],
                    params['policy']['class'],
                    params['policy']['H'],
                    params['policy']['use_target'],
                    params['policy']['RCcarMACPolicy']['is_classification'] if params['policy']['class'] == 'RCcarMACPolicy' else '',
                    params['policy']['RCcarMACPolicy']['probcoll_strictly_increasing'] if params['policy']['class'] == 'RCcarMACPolicy' else '',
                    params['alg']['replay_pool_sampling'],
                    params['policy']['clip_cost_target_with_dones'],
                )

                ax.set_title(title, fontdict={'fontsize': 5})

    f_cumreward.savefig(os.path.join(SAVE_FOLDER, '{0}_cumreward.png'.format(FILE_NAME)), bbox_inches='tight', dpi=150)
    plt.close(f_cumreward)

def plot_2698_2757_and_2868_2891():
    FILE_NAME = 'rccar_2698_2757_and_2868_2891'

    all_exps = [load_experiments(range(i, i + 3), load_eval_rollouts=False) for i in range(2698, 2757, 3)]
    all_exps = all_exps[:4] + \
               [load_experiments(range(i, i + 3), load_eval_rollouts=False) for i in range(2868, 2891, 3)] + \
               all_exps[4:]

    # import IPython; IPython.embed()

    f_cumreward, axes_cumreward = plt.subplots(7, 4, figsize=(12, 21), sharey=True, sharex=True)

    window = 16
    ylim = (0, 2100)
    success_cumreward = [500, 1000, 1500, 1750]

    for ax_cumreward, exp in zip(axes_cumreward.ravel(), all_exps):

        if len(exp) > 0:
            try:
                plot_cumreward(ax_cumreward, exp, window=window, success_cumreward=success_cumreward, ylim=ylim)
            except:
                pass
            params = exp[0].params
            for ax in (ax_cumreward,):
                title = '{0}, {1}, {2}, H: {3},\ntarg: {4}, classif: {5}, incr: {6}, rp: {7}, clip: {8}'.format(
                    params['exp_name'],
                    params['alg']['env'].split('(params=')[0].split('"')[-1],
                    params['policy']['class'],
                    params['policy']['H'],
                    params['policy']['use_target'],
                    params['policy']['RCcarMACPolicy']['is_classification'] if params['policy']['class'] == 'RCcarMACPolicy' else '',
                    params['policy']['RCcarMACPolicy']['probcoll_strictly_increasing'] if params['policy']['class'] == 'RCcarMACPolicy' else '',
                    params['alg']['replay_pool_sampling'],
                    params['policy']['clip_cost_target_with_dones'],
                )

                ax.set_title(title, fontdict={'fontsize': 5})

    f_cumreward.savefig(os.path.join(SAVE_FOLDER, '{0}_cumreward.png'.format(FILE_NAME)), bbox_inches='tight', dpi=150)
    plt.close(f_cumreward)

def plot_2759_2794():
    FILE_NAME = 'rccar_2759_2794'

    all_exps = [load_experiments(range(i, i + 3), load_eval_rollouts=False) for i in range(2759, 2794, 3)]

    # import IPython; IPython.embed()

    f_cumreward, axes_cumreward = plt.subplots(4, 3, figsize=(9, 12), sharey=True, sharex=False)

    window = 16
    ylim = (0, 2100)
    success_cumreward = [500, 1000, 1500, 1750]

    for ax_cumreward, exp in zip(axes_cumreward.ravel(), all_exps):

        if len(exp) > 0:
            try:
                plot_cumreward(ax_cumreward, exp, window=window, success_cumreward=success_cumreward, ylim=ylim)
            except:
                pass
            params = exp[0].params
            for ax in (ax_cumreward,):
                title = '{0}, {1}, {2}, backup: {3},\nN: {4}, H: {5}, rp: {6}'.format(
                    params['exp_name'],
                    params['alg']['env'].split('(params=')[0].split('"')[-1],
                    params['alg']['env'].split("'do_back_up':")[1].split(',')[0],
                    params['policy']['class'],
                    params['policy']['N'],
                    params['policy']['H'],
                    params['alg']['replay_pool_sampling'],
                )

                ax.set_title(title, fontdict={'fontsize': 5})

    for i, xmax in enumerate([2e5, 4e5, 4e5, 8e5]):
        for ax in axes_cumreward[i, :]:
            ax.set_xlim((0, xmax))
            for tick in ax.xaxis.get_major_ticks():
                tick.label.set_fontsize(6)
                tick.set_visible(True)

    f_cumreward.savefig(os.path.join(SAVE_FOLDER, '{0}_cumreward.png'.format(FILE_NAME)), bbox_inches='tight', dpi=150)
    plt.close(f_cumreward)

def plot_2893_3084():
    FILE_NAME = 'rccar_2893_3084'

    all_exps = np.array([load_experiments(range(i, i + 3), load_eval_rollouts=False) for i in range(2893, 3084, 3)])

    window = 16
    ylim = (0, 2100)
    success_cumreward = [500, 1000, 1500, 1750]
    xmaxes = (2e5, 4e5, 4e5, 8e5)

    for i, exps in enumerate(np.split(all_exps, 4)):

        f_cumreward, axes_cumreward = plt.subplots(4, 4, figsize=(12, 12), sharey=True, sharex=False)

        for ax_cumreward, exp in zip(axes_cumreward.ravel(), exps):

            if not hasattr(exp, '__len__'):
                exp = [exp]

            if len(exp) > 0:
                try:
                    plot_cumreward(ax_cumreward, exp, window=window, success_cumreward=success_cumreward, ylim=ylim,
                                   xmax=xmaxes[i])
                except:
                    pass
                params = exp[0].params
                title = '{0}, {1}, backup: {2},\nH: {3}, H targ: {4}, incr: {5}, clip: {6}'.format(
                    params['exp_name'],
                    params['alg']['env'].split('(params=')[0].split('"')[-1],
                    params['alg']['env'].split("'do_back_up':")[1].split(',')[0],
                    params['policy']['H'],
                    params['policy']['get_action_target']['H'],
                    params['policy']['RCcarMACPolicy']['probcoll_strictly_increasing'],
                    params['policy']['clip_cost_target_with_dones']
                )
                ax_cumreward.set_title(title, fontdict={'fontsize': 6})

        for ax in axes_cumreward.ravel():
            ax.set_xlim((0, xmaxes[i]))
            for tick in ax.xaxis.get_major_ticks():
                tick.label.set_fontsize(6)

        f_cumreward.savefig(os.path.join(SAVE_FOLDER, '{0}_cumreward_{1}.png'.format(FILE_NAME, i)),
                            bbox_inches='tight',
                            dpi=200)
        plt.close(f_cumreward)

###################
### Final plots ###
###################

FINAL_DEBUG = False
font = {'family' : 'serif',
        'weight': 'normal',
        'size'   : 12}
matplotlib.rc('font', **font)
# matplotlib.rc('text', usetex=True)

def plot_env_type(FILE_NAME, all_exps, titles, xmax_timesteps, show_title, show_xlabel, plot_title, xtext=0):
    f_cumreward, axes_cumreward = plt.subplots(1, len(all_exps), figsize=(2*len(all_exps), 2), sharey=False, sharex=True)

    window = 16
    ylim = (0, 2100)
    success_cumreward = [500, 1000, 1500, 1750]

    for ax_cumreward, exp, title in zip(axes_cumreward.ravel(), all_exps, titles):

        if len(exp) > 0:
            plot_cumreward(ax_cumreward, exp, window=window, success_cumreward=success_cumreward, ylim=ylim,
                           plot_indiv=False, convert_to_time=True)
            params = exp[0].params

            if FINAL_DEBUG:
                title = '{0}, {1}, {2}, backup: {3},\nN: {4}, H: {5}, rp: {6}'.format(
                    params['exp_name'],
                    params['alg']['env'].split('(params=')[0].split('"')[-1],
                    params['alg']['env'].split("'do_back_up':")[1].split(',')[0],
                    params['policy']['class'],
                    params['policy']['N'],
                    params['policy']['H'],
                    params['alg']['replay_pool_sampling'],
                    )

            if show_title:
                ax_cumreward.set_title(title, fontdict=font)#, fontdict={'fontsize': 5})

    # set same x-axis
    xmax = xmax_timesteps * (0.25 / 3600.)
    for ax in axes_cumreward:
        ax.set_ylim(ylim)
        ax.set_xlim((0, xmax))
        ax.yaxis.set_ticks(np.arange(0, 2100, 500))
        ax.set_yticklabels([''] * len(ax.get_yticklabels()))
        ax.yaxis.set_ticks_position('both')
        # for tick in ax.xaxis.get_major_ticks():
        #     tick.label.set_fontsize(6)

    if show_xlabel:
        f_cumreward.text(0.5, -0.12, 'Time (hours)', ha='center', fontdict=font)

    ax = axes_cumreward[0]
    ax.set_yticklabels(['', '250', '500', '750', '1000', ''])
    ax.set_ylabel('Distance (m)', fontdict=font)

    ax = axes_cumreward[-1]
    ax.yaxis.tick_right()
    ax.set_yticklabels(['', '3', '6', '9', '12', ''])
    ax.set_ylabel('Hallway lengths', fontdict=font)
    ax.yaxis.set_label_position("right")

    # add y-axis on right side which is hallway lengths
    if plot_title:
        f_cumreward.text(xtext, 0.5, plot_title, va='center', ha='center', rotation=90, fontdict=font)

    f_cumreward.savefig(os.path.join(SAVE_FOLDER, '{0}.png'.format(FILE_NAME)), bbox_inches='tight', dpi=200)
    plt.close(f_cumreward)

def plot_empty_hallway_reset():
    FILE_NAME = 'rccar_paper_empty_hallway_reset'
    # all_exps = [load_experiments(range(i, i + 3), load_eval_rollouts=False) for i in range(2445, 2463, 3)]
    all_exps = [load_experiments(range(i, i + 3), load_eval_rollouts=False) for i in range(2445, 2460, 3)] + \
               [load_experiments(range(2557, 2557 + 3), load_eval_rollouts=False)]
    titles = ['Double\nQ-learning', '5-step Double\nQ-learning', '10-step Double\nQ-learning', '5-step MAQL',
              '10-step MAQL', 'Collision\nPrediction (ours)']

    plot_env_type(FILE_NAME, all_exps, titles, xmax_timesteps=2e5, show_title=True, show_xlabel=False,
                  plot_title='(a) Empty hallway\n(reset)')

def plot_empty_hallway_lifelong():
    FILE_NAME = 'rccar_paper_empty_hallway_lifelong'
    # all_exps = [load_experiments(range(i, i + 3), load_eval_rollouts=False) for i in range(2463, 2481, 3)]
    all_exps = [load_experiments(range(i, i + 3), load_eval_rollouts=False) for i in range(2463, 2478, 3)] + \
               [load_experiments(range(2617, 2617 + 3), load_eval_rollouts=False)]
    titles = ['Double\nQ-learning', '5-step Double\nQ-learning', '10-step Double\nQ-learning', '5-step MAQL',
              '10-step MAQL', 'Collision\nPrediction (ours)']

    plot_env_type(FILE_NAME, all_exps, titles, xmax_timesteps=4e5, show_title=False, show_xlabel=False,
                  plot_title='(b) Empty hallway\n(lifelong)')

def plot_cluttered_hallway_reset():
    FILE_NAME = 'rccar_paper_cluttered_hallway_reset'
    # all_exps = [load_experiments(range(i, i + 3), load_eval_rollouts=False) for i in range(2481, 2499, 3)]
    all_exps = [load_experiments(range(i, i + 3), load_eval_rollouts=False) for i in range(2481, 2496, 3)] + \
               [load_experiments(range(2677, 2677 + 3), load_eval_rollouts=False)]
    all_exps[1] = [all_exps[1][0], all_exps[1][2]]
    titles = ['Double\nQ-learning', '5-step Double\nQ-learning', '10-step Double\nQ-learning', '5-step MAQL',
              '10-step MAQL', 'Collision\nPrediction (ours)']

    plot_env_type(FILE_NAME, all_exps, titles, xmax_timesteps=4e5, show_title=False, show_xlabel=False,
                  plot_title='(c) Cluttered hallway\n(reset)')

def plot_cluttered_hallway_lifelong():
    FILE_NAME = 'rccar_paper_cluttered_hallway_lifelong'
    # all_exps = [load_experiments(range(i, i + 3), load_eval_rollouts=False) for i in range(2499, 2517, 3)]
    all_exps = [load_experiments(range(i, i + 3), load_eval_rollouts=False) for i in range(2499, 2514, 3)] + \
               [load_experiments(range(2737, 2737 + 3), load_eval_rollouts=False)]
    titles = ['Double\nQ-learning', '5-step Double\nQ-learning', '10-step Double\nQ-learning', '5-step MAQL',
              '10-step MAQL', 'Collision\nPrediction (ours)']

    plot_env_type(FILE_NAME, all_exps, titles, xmax_timesteps=8e5, show_title=False, show_xlabel=True,
                  plot_title='(d) Cluttered hallway\n(lifelong)')

def plot_priority_replay_empty_hallway_reset():
    FILE_NAME = 'rccar_paper_priority_replay_empty_hallway_reset'
    all_exps = [load_experiments(range(2759, 2759 + 3), load_eval_rollouts=False),
                load_experiments(range(2762, 2762 + 3), load_eval_rollouts=False),
                load_experiments(range(2765, 2765 + 3), load_eval_rollouts=False),
                load_experiments(range(2563, 2563+ 3), load_eval_rollouts=False)]
    titles = ['Double\nQ-learning', '5-step Double\nQ-learning', '10-step Double\nQ-learning', 'Collision\nPrediction (ours)']

    plot_env_type(FILE_NAME, all_exps, titles, xmax_timesteps=2e5, show_title=True, show_xlabel=False,
                  plot_title='(a) Empty hallway\n(reset)', xtext=-0.05)

def plot_priority_replay_empty_hallway_lifelong():
    FILE_NAME = 'rccar_paper_priority_replay_empty_hallway_lifelong'
    all_exps = [load_experiments(range(2768, 2768 + 3), load_eval_rollouts=False),
                load_experiments(range(2771, 2771 + 3), load_eval_rollouts=False),
                load_experiments(range(2774, 2774 + 3), load_eval_rollouts=False),
                load_experiments(range(2623, 2623 + 3), load_eval_rollouts=False)]
    titles = ['Double\nQ-learning', '5-step Double\nQ-learning', '10-step Double\nQ-learning', 'Collision\nPrediction (ours)']

    plot_env_type(FILE_NAME, all_exps, titles, xmax_timesteps=4e5, show_title=False, show_xlabel=False,
                  plot_title='(b) Empty hallway\n(lifelong)', xtext=-0.05)

def plot_priority_replay_cluttered_hallway_reset():
    FILE_NAME = 'rccar_paper_priority_replay_cluttered_hallway_reset'
    all_exps = [load_experiments(range(2777, 2777 + 3), load_eval_rollouts=False),
                load_experiments(range(2780, 2780 + 3), load_eval_rollouts=False),
                load_experiments(range(2783, 2783 + 3), load_eval_rollouts=False),
                load_experiments(range(2683, 2683 + 3), load_eval_rollouts=False)]
    titles = ['Double\nQ-learning', '5-step Double\nQ-learning', '10-step Double\nQ-learning', 'Collision\nPrediction (ours)']

    plot_env_type(FILE_NAME, all_exps, titles, xmax_timesteps=4e5, show_title=False, show_xlabel=False,
                  plot_title='(c) Cluttered hallway\n(reset)', xtext=-0.05)

def plot_priority_replay_cluttered_hallway_lifelong():
    FILE_NAME = 'rccar_paper_priority_replay_cluttered_hallway_lifelong'
    all_exps = [load_experiments(range(2786, 2786 + 3), load_eval_rollouts=False),
                load_experiments(range(2789, 2789 + 3), load_eval_rollouts=False),
                load_experiments(range(2792, 2792 + 3), load_eval_rollouts=False),
                load_experiments(range(2743, 2743 + 3), load_eval_rollouts=False)]
    titles = ['Double\nQ-learning', '5-step Double\nQ-learning', '10-step Double\nQ-learning', 'Collision\nPrediction (ours)']

    plot_env_type(FILE_NAME, all_exps, titles, xmax_timesteps=8e5, show_title=False, show_xlabel=True,
                  plot_title='(d) Cluttered hallway\n(lifelong)', xtext=-0.05)

def plot_design_decisions(FILE_NAME, all_exps):
    assert(len(all_exps) == 28)
    f_cumreward, axes_cumreward = plt.subplots(7, 4, figsize=(8, 14), sharey=True, sharex=True)

    window = 16
    ylim = (0, 2100)
    success_cumreward = [500, 1000, 1500, 1750]

    for ax_cumreward, exp in zip(axes_cumreward.ravel(), all_exps):

        if len(exp) > 0:
            try:
                plot_cumreward(ax_cumreward, exp, window=window, success_cumreward=success_cumreward, ylim=ylim,
                               plot_indiv=False, convert_to_time=True)
            except:
                pass

            params = exp[0].params
            if FINAL_DEBUG:
                title = '{0}, {1}, {2}, H: {3},\ntarg: {4}, classif: {5}, incr: {6}, rp: {7}, clip: {8}'.format(
                    params['exp_name'],
                    params['alg']['env'].split('(params=')[0].split('"')[-1],
                    params['policy']['class'],
                    params['policy']['H'],
                    params['policy']['use_target'],
                    params['policy']['RCcarMACPolicy']['is_classification'] if params['policy']['class'] == 'RCcarMACPolicy' else '',
                    params['policy']['RCcarMACPolicy']['probcoll_strictly_increasing'] if params['policy']['class'] == 'RCcarMACPolicy' else '',
                    params['alg']['replay_pool_sampling'],
                    params['policy']['clip_cost_target_with_dones'],
                )
                fontsize = 5
            else:
                title = ''
                if params['policy']['class'] == 'MACPolicy':
                    if params['policy']['use_target']:
                        title += 'MAQL, H: {0}, H target: {1}'.format(
                            params['policy']['H'],
                            params['policy']['get_action_target']['H']
                        )
                    else:
                        title += 'MAQL, H: {0}, no target'.format(
                            params['policy']['H'],
                        )
                elif params['policy']['class'] == 'RCcarMACPolicy':
                    title += 'Probcoll, H: {0}'.format(
                        params['policy']['H']
                    )
                else:
                    raise NotImplementedError

                if params['policy']['class'] == 'RCcarMACPolicy':
                    title += ', {0}, {1}'.format(
                        'classification' if params['policy']['RCcarMACPolicy']['is_classification'] else 'regression',
                        'strictly increasing' if params['policy']['RCcarMACPolicy']['probcoll_strictly_increasing'] else 'not strictly increasing'
                    )

                title += ',\n{0}, {1}'.format(
                    'uniform replay' if params['alg']['replay_pool_sampling'] == 'uniform' else 'prioritized replay',
                    'clip cost' if params['policy']['clip_cost_target_with_dones'] else "don't clip cost"
                )

                fontsize = 8

            ax_cumreward.set_title(title, fontdict={'fontsize': fontsize})

    # set same x-axis
    for ax in axes_cumreward.ravel():
        ax.set_ylim(ylim)
        ax.yaxis.set_ticks(np.arange(0, 2100, 500))
        ax.set_yticklabels([''] * len(ax.get_yticklabels()))
        ax.yaxis.set_ticks_position('both')

    for ax in axes_cumreward[:, 0]:
        ax.set_yticklabels(['', '250', '500', '750', '1000', ''])
        ax.set_ylabel('Distance (m)', fontdict=font)

    for ax in axes_cumreward[:, -1]:
        ax.yaxis.tick_right()
        ax.set_yticklabels(['', '3', '6', '9', '12', ''])
        ax.set_ylabel('Hallway lengths', fontdict=font)
        ax.yaxis.set_label_position("right")

    f_cumreward.text(0.5, -0.04, 'Time (hours)', ha='center', fontdict=font)

    f_cumreward.subplots_adjust(top=1.1, bottom=0.0, left=-0.2, right=1.2)

    f_cumreward.savefig(os.path.join(SAVE_FOLDER, '{0}.png'.format(FILE_NAME)), bbox_inches='tight', dpi=150)
    plt.close(f_cumreward)

def plot_design_decisions_empty_hallway_reset():
    FILE_NAME = 'rccar_paper_design_decisions_empty_hallway_reset'

    all_exps = [load_experiments(range(i, i + 3), load_eval_rollouts=False) for i in range(2518, 2577, 3)]
    all_exps = all_exps[:4] + \
               [load_experiments(range(i, i + 3), load_eval_rollouts=False) for i in range(2796, 2819, 3)] + \
               all_exps[4:]

    plot_design_decisions(FILE_NAME, all_exps)

def plot_design_decisions_empty_hallway_lifelong():
    FILE_NAME = 'rccar_paper_design_decisions_empty_hallway_lifelong'

    all_exps = [load_experiments(range(i, i + 3), load_eval_rollouts=False) for i in range(2578, 2637, 3)]
    all_exps = all_exps[:4] + \
               [load_experiments(range(i, i + 3), load_eval_rollouts=False) for i in range(2820, 2843, 3)] + \
               all_exps[4:]

    plot_design_decisions(FILE_NAME, all_exps)

def plot_design_decisions_cluttered_hallway_reset():
    FILE_NAME = 'rccar_paper_design_decisions_cluttered_hallway_reset'

    all_exps = [load_experiments(range(i, i + 3), load_eval_rollouts=False) for i in range(2638, 2697, 3)]
    all_exps = all_exps[:4] + \
               [load_experiments(range(i, i + 3), load_eval_rollouts=False) for i in range(2844, 2867, 3)] + \
               all_exps[4:]

    plot_design_decisions(FILE_NAME, all_exps)

def plot_design_decisions_cluttered_hallway_lifelong():
    FILE_NAME = 'rccar_paper_design_decisions_cluttered_hallway_lifelong'

    all_exps = [load_experiments(range(i, i + 3), load_eval_rollouts=False) for i in range(2698, 2757, 3)]
    all_exps = all_exps[:4] + \
               [load_experiments(range(i, i + 3), load_eval_rollouts=False) for i in range(2868, 2891, 3)] + \
               all_exps[4:]

    plot_design_decisions(FILE_NAME, all_exps)

def plot_baselines_prioritized_replay():
    pass

def plot_dd_heatmap(FILE_NAME, exp_nums, xmax):
    all_exps = [load_experiments(range(i, i + 3), load_eval_rollouts=False) for i in exp_nums[::-1]]

    window = 16
    success_cumreward = [500, 1000, 1500, 1750]
    convert_to_time = True

    if convert_to_time:
        xmax *= 0.25 / 3600. # times dt and to hours

    ### get thresholds
    all_threshold_steps = []
    all_final_step = []
    all_final_value = []
    all_names = []
    for exps in all_exps:
        threshold_steps, final_step, final_value = \
            get_threshold_steps_and_final_performance(exps, success_cumreward,
                                                      convert_to_time=convert_to_time, window=window)

        all_threshold_steps.append(threshold_steps)
        all_final_step.append(final_step)
        all_final_value.append(final_value)

        params = exps[0].params
        # output_name = 'value' if params['policy']['class'] == 'MACPolicy' else 'collision'
        # bootstrap_name = 'bootstrap' if params['policy']['use_target'] else 'no bootstrap'
        # if params['policy']['class'] == 'MACPolicy':
        #     loss_name = 'regression'
        # else:
        #     if params['policy']['RCcarMACPolicy']['is_classification']:
        #         loss_name = 'classification'
        #     else:
        #         loss_name = 'regression'
        # horizon_name = 'short horizon' if params['policy']['H'] == 1 else 'long horizon'
        output_name = 'v' if params['policy']['class'] == 'MACPolicy' else 'c'
        bootstrap_name = '+b' if params['policy']['use_target'] else '-b'
        if params['policy']['class'] == 'MACPolicy':
            loss_name = 'r'
        else:
            if params['policy']['RCcarMACPolicy']['is_classification']:
                loss_name = 'c'
            else:
                loss_name = 'r'
        horizon_name = 'h' if params['policy']['H'] == 1 else 'H'
        name = '{0}, {1}, {2}, {3}'.format(output_name, bootstrap_name, loss_name, horizon_name)
        all_names.append(name)


    all_threshold_steps = np.array(all_threshold_steps)
    all_final_step = np.array(all_final_step)
    all_final_value = np.array([all_final_value]).T / 2.

    all_threshold_steps[all_threshold_steps < 0] = xmax

    assert (all_threshold_steps.min() >= 0)
    assert (all_threshold_steps.max() <= xmax)

    ### plot heatmaps
    f = plt.figure(figsize=(8, 3))
    gs = matplotlib.gridspec.GridSpec(1, 2, width_ratios=[4, 1])
    ax_thresh = plt.subplot(gs[0])
    ax_fv = plt.subplot(gs[1])

    ### plot thresholds
    heatmap = ax_thresh.pcolor(all_threshold_steps, cmap=cm.magma_r)

    p0 = ax_thresh.get_position().get_points().flatten()
    ax_cbar = f.add_axes([p0[0], 0.9, p0[2] - p0[0], 0.05])
    cbar = plt.colorbar(heatmap, cax=ax_cbar, orientation='horizontal')

    for y in range(all_threshold_steps.shape[0]):
        for x in range(all_threshold_steps.shape[1]):
            color = 'k' if all_threshold_steps[y, x] < 20 else 'w'
            if all_threshold_steps[y, x] < xmax:
                value = '%.1f' % all_threshold_steps[y, x]
            else:
                value = 'N/A'
            ax_thresh.text(x + 0.5, y + 0.5, value,
                     horizontalalignment='center',
                     verticalalignment='center',
                           color=color
                     )

    cbar.ax.xaxis.set_ticks_position('top')
    cbar.ax.xaxis.set_label_position('top')
    cbar.set_label('Hours', labelpad=10, y=1.05, rotation=0)

    ax_thresh.set_xticks(np.r_[0.5:len(success_cumreward):1])
    ax_thresh.set_xticklabels(['{0}m'.format(sc // 2) for sc in success_cumreward])
    ax_thresh.set_yticks(np.r_[0.5:len(all_names):1])
    ax_thresh.set_yticklabels(all_names)

    ### plot final values
    heatmap = ax_fv.pcolor(all_final_value, vmin=0., vmax=1000., cmap=cm.magma)

    p0 = ax_fv.get_position().get_points().flatten()
    ax_cbar = f.add_axes([p0[0] + 0.15, p0[1], 0.04, p0[3] - p0[1]])
    cbar = plt.colorbar(heatmap, cax=ax_cbar)

    for y in range(all_final_value.shape[0]):
        for x in range(all_final_value.shape[1]):
            color = 'k' if all_final_value[y, x] > 300 else 'w'
            ax_fv.text(x + 0.5, y + 0.5, '%d' % all_final_value[y, x],
                     horizontalalignment='center',
                     verticalalignment='center',
                       color=color
                     )

    ax_fv.set_title('Final performance (m)', loc='left')
    for ax in (ax_fv,):
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)

    f.savefig(os.path.join(SAVE_FOLDER, '{0}.png'.format(FILE_NAME)), bbox_inches='tight', dpi=150)
    plt.close(f)

def plot_dd_heatmap_empty_hallway_lifelong():
    FILE_NAME = 'rccar_paper_dd_heatmap_empty_hallway_lifelong'

    xmax = 4e5
    exp_nums = [2463, 2472, 2835, 2593, 3046, 3082, 2617]

    plot_dd_heatmap(FILE_NAME, exp_nums, xmax)

def plot_dd_heatmap_cluttered_hallway_lifelong():
    FILE_NAME = 'rccar_paper_dd_heatmap_cluttered_hallway_lifelong'

    xmax = 8e5
    exp_nums = [2499, 2508, 2883, 2713, 3046, 3082, 2737]

    plot_dd_heatmap(FILE_NAME, exp_nums, xmax)

def plot_dd_cluttered_hallway_lifelong_outputs_loss():
    # \textbf{Model outputs and loss function.}
    # % compare:
    # %   value + reg         (rccar2883)
    # %   collision + reg     (rccar2713)
    # %   collision + classif (rccar2737)
    # % constants: no bootstrap, long horizon
    font = {'family': 'serif',
            'weight': 'normal',
            'size': 12}
    matplotlib.rc('font', **font)

    FILE_NAME = 'rccar_paper_dd_cluttered_hallway_lifelong_outputs_loss'

    exp_nums = (2883, 2713, 2737)
    all_exps = [load_experiments(range(i, i + 3), load_eval_rollouts=False) for i in exp_nums]
    titles = ['value, regression', 'collision, regression', 'collision, classification']
    colors = ['k', 'r', 'g']

    f_cumreward, axes_cumreward = plt.subplots(1, 1, figsize=(4, 3), sharey=False, sharex=True)
    if not hasattr(axes_cumreward, '__len__'):
        axes_cumreward = np.array([axes_cumreward])

    window = 32
    ylim = (0, 2100)
    xmax_timesteps = 8e5

    for exp, title, color in zip(all_exps[::-1], titles[::-1], colors[::-1]):
        # params = exp[0].params
        # title = '{0}, {1}, {2}, backup: {3},\nN: {4}, H: {5}, rp: {6}'.format(
        #     params['exp_name'],
        #     params['alg']['env'].split('(params=')[0].split('"')[-1],
        #     params['alg']['env'].split("'do_back_up':")[1].split(',')[0],
        #     params['policy']['class'],
        #     params['policy']['N'],
        #     params['policy']['H'],
        #     params['alg']['replay_pool_sampling'],
        # )

        plot_cumreward(axes_cumreward[0], exp, window=window, ylim=ylim, plot_indiv=False, convert_to_time=True, label=title, color=color)

    # set same x-axis
    xmax = xmax_timesteps * (0.25 / 3600.)
    for ax in axes_cumreward:
        ax.set_ylim(ylim)
        ax.set_xlim((0, xmax))
        ax.yaxis.set_ticks(np.arange(0, 2100, 500))
        ax.set_yticklabels([''] * len(ax.get_yticklabels()))
        ax.yaxis.set_ticks_position('both')

        ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.4))

    f_cumreward.text(0.5, -0.05, 'Time (hours)', ha='center', fontdict=font)

    ax = axes_cumreward[0]
    ax.set_yticklabels(['', '250', '500', '750', '1000', ''])
    ax.set_ylabel('Distance (m)', fontdict=font)

    # add y-axis on right side which is hallway lengths
    ax = axes_cumreward[-1]
    ax_twin = ax.twinx()
    ax_twin.set_xlim((ax.get_xlim()))
    ax_twin.set_ylim((ax.get_ylim()))
    ax_twin.set_yticks(ax.get_yticks())
    ax_twin.yaxis.tick_right()
    ax_twin.set_yticklabels(['', '3', '6', '9', '12', ''])
    ax_twin.set_ylabel('Hallway lengths', fontdict=font)
    ax_twin.yaxis.set_label_position("right")

    f_cumreward.savefig(os.path.join(SAVE_FOLDER, '{0}.png'.format(FILE_NAME)), bbox_inches='tight', dpi=200)
    plt.close(f_cumreward)

def plot_dd_cluttered_hallway_lifelong_horizon():
    # \textbf{Model horizon.}
    # % compare:
    # %   value short horizon (rccar2499)
    # %   value long horizon  (rccar2508)
    # %   coll short horizon  (rccar3046)
    # %   coll long horizon   (rccar3082)
    # % constants: yes bootstrapping, reg for value and classif for coll
    font = {'family': 'serif',
            'weight': 'normal',
            'size': 15}
    matplotlib.rc('font', **font)

    FILE_NAME = 'rccar_paper_dd_cluttered_hallway_lifelong_horizon'

    exp_nums = (2499, 2508, 3046, 3082)
    all_exps = [load_experiments(range(i, i + 3), load_eval_rollouts=False) for i in exp_nums]
    titles = ['Value', 'Value', 'Collision', 'Collision']
    labels = ['short horizon', 'long horizon'] * 2
    colors = ['m', 'r', 'b', 'c']

    f_cumreward, axes_cumreward = plt.subplots(1, 2, figsize=(8, 3), sharey=False, sharex=True)
    axes_cumreward = np.array([axes_cumreward[0], axes_cumreward[0], axes_cumreward[1], axes_cumreward[1]])

    window = 32
    ylim = (0, 2100)
    xmax_timesteps = 8e5
    xmax = xmax_timesteps * (0.25 / 3600.)

    for ax, exp, title, label, color in zip(axes_cumreward.ravel(), all_exps, titles, labels, colors):
        # params = exp[0].params
        # title = '{0}, {1}, {2}, backup: {3},\nN: {4}, H: {5}, rp: {6}'.format(
        #     params['exp_name'],
        #     params['alg']['env'].split('(params=')[0].split('"')[-1],
        #     params['alg']['env'].split("'do_back_up':")[1].split(',')[0],
        #     params['policy']['class'],
        #     params['policy']['N'],
        #     params['policy']['H'],
        #     params['alg']['replay_pool_sampling'],
        # )

        plot_cumreward(ax, exp, window=window, ylim=ylim, plot_indiv=False, convert_to_time=True, label=label, color=color,
                       xmax=xmax)
        ax.set_title(title)

    # set same x-axis
    for ax in axes_cumreward:
        ax.set_ylim(ylim)
        ax.set_xlim((0, xmax))
        ax.yaxis.set_ticks(np.arange(0, 2100, 500))
        ax.set_yticklabels([''] * len(ax.get_yticklabels()))
        ax.yaxis.set_ticks_position('both')

        # ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.4))

    # patch0 = mpatches.Patch(color='red', lw=0.1)
    # patch0 = mpatches.Arrow(0, 0, 0.1, 0, color='blue', width=0.5)
    patches = []
    patches += [mlines.Line2D([], [], linewidth=3., color=colors[1])]
    patches += [mlines.Line2D([], [], linewidth=3., color=colors[0])]
    patches += [mlines.Line2D([], [], linewidth=3., color=colors[3])]
    patches += [mlines.Line2D([], [], linewidth=3., color=colors[2])]

    leg_labels = ['', '', labels[1], labels[0]]
    f_cumreward.legend(ncol=2, handles=patches, labels=leg_labels, loc='upper center', bbox_to_anchor=(0.5, 1.4))

    f_cumreward.text(0.5, -0.05, 'Time (hours)', ha='center', fontdict=font)

    ax = axes_cumreward[0]
    ax.set_yticklabels(['', '250', '500', '750', '1000', ''])
    ax.set_ylabel('Distance (m)', fontdict=font)

    # add y-axis on right side which is hallway lengths
    ax = axes_cumreward[-1]
    ax_twin = ax.twinx()
    ax_twin.set_xlim((axes_cumreward[0].get_xlim()))
    ax_twin.set_ylim((axes_cumreward[0].get_ylim()))
    ax_twin.set_yticks(ax.get_yticks())
    ax_twin.yaxis.tick_right()
    ax_twin.set_yticklabels(['', '3', '6', '9', '12', ''])
    ax_twin.set_ylabel('Hallway lengths', fontdict=font)
    ax_twin.yaxis.set_label_position("right")

    f_cumreward.savefig(os.path.join(SAVE_FOLDER, '{0}.png'.format(FILE_NAME)), bbox_inches='tight', dpi=200)
    plt.close(f_cumreward)

def plot_dd_cluttered_hallway_lifelong_bootstrapping():
    # \textbf{Bootstrapping.}
    # % compare:
    # %   value -b (rccar2883)
    # %   value +b (rccar2508)
    # %   coll -b  (rccar2737)
    # %   coll +b  (rccar3082)
    # % constants: horizon (long), reg for value and classif for coll
    font = {'family': 'serif',
            'weight': 'normal',
            'size': 15}
    matplotlib.rc('font', **font)

    FILE_NAME = 'rccar_paper_dd_cluttered_hallway_lifelong_bootstrapping'

    exp_nums = (2883, 2508, 2737, 3082)
    all_exps = [load_experiments(range(i, i + 3), load_eval_rollouts=False) for i in exp_nums]
    titles = ['Value', 'Value', 'Collision', 'Collision']
    labels = ['no bootstrap', 'bootstrap'] * 2
    colors = ['m', 'r', 'b', 'c']

    f_cumreward, axes_cumreward = plt.subplots(1, 2, figsize=(8, 3), sharey=False, sharex=True)
    axes_cumreward = np.array([axes_cumreward[0], axes_cumreward[0], axes_cumreward[1], axes_cumreward[1]])

    window = 32
    ylim = (0, 2100)
    xmax_timesteps = 8e5
    xmax = xmax_timesteps * (0.25 / 3600.)

    for ax, exp, title, label, color in zip(axes_cumreward.ravel(), all_exps, titles, labels, colors):
        # params = exp[0].params
        # title = '{0}, {1}, {2}, backup: {3},\nN: {4}, H: {5}, rp: {6}'.format(
        #     params['exp_name'],
        #     params['alg']['env'].split('(params=')[0].split('"')[-1],
        #     params['alg']['env'].split("'do_back_up':")[1].split(',')[0],
        #     params['policy']['class'],
        #     params['policy']['N'],
        #     params['policy']['H'],
        #     params['alg']['replay_pool_sampling'],
        # )

        plot_cumreward(ax, exp, window=window, ylim=ylim, plot_indiv=False, convert_to_time=True, label=label, color=color,
                       xmax=xmax)
        ax.set_title(title)

    # set same x-axis
    for ax in axes_cumreward:
        ax.set_ylim(ylim)
        ax.set_xlim((0, xmax))
        ax.yaxis.set_ticks(np.arange(0, 2100, 500))
        ax.set_yticklabels([''] * len(ax.get_yticklabels()))
        ax.yaxis.set_ticks_position('both')

        # ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.4))

    # patch0 = mpatches.Patch(color='red', lw=0.1)
    # patch0 = mpatches.Arrow(0, 0, 0.1, 0, color='blue', width=0.5)
    patches = []
    patches += [mlines.Line2D([], [], linewidth=3., color=colors[1])]
    patches += [mlines.Line2D([], [], linewidth=3., color=colors[0])]
    patches += [mlines.Line2D([], [], linewidth=3., color=colors[3])]
    patches += [mlines.Line2D([], [], linewidth=3., color=colors[2])]

    leg_labels = ['', '', labels[1], labels[0]]
    f_cumreward.legend(ncol=2, handles=patches, labels=leg_labels, loc='upper center', bbox_to_anchor=(0.5, 1.4))

    f_cumreward.text(0.5, -0.05, 'Time (hours)', ha='center', fontdict=font)

    ax = axes_cumreward[0]
    ax.set_yticklabels(['', '250', '500', '750', '1000', ''])
    ax.set_ylabel('Distance (m)', fontdict=font)

    # add y-axis on right side which is hallway lengths
    ax = axes_cumreward[-1]
    ax_twin = ax.twinx()
    ax_twin.set_xlim((axes_cumreward[0].get_xlim()))
    ax_twin.set_ylim((axes_cumreward[0].get_ylim()))
    ax_twin.set_yticks(ax.get_yticks())
    ax_twin.yaxis.tick_right()
    ax_twin.set_yticklabels(['', '3', '6', '9', '12', ''])
    ax_twin.set_ylabel('Hallway lengths', fontdict=font)
    ax_twin.yaxis.set_label_position("right")

    f_cumreward.savefig(os.path.join(SAVE_FOLDER, '{0}.png'.format(FILE_NAME)), bbox_inches='tight', dpi=200)
    plt.close(f_cumreward)

def plot_cluttered_hallway_lifelong_standalone():
    font = {'family': 'serif',
            'weight': 'normal',
            'size': 15}
    matplotlib.rc('font', **font)

    FILE_NAME = 'rccar_paper_cluttered_hallway_lifelong_standalone'
    exp_nums = (2499, 2499 + 3, 2737)
    all_exps = [load_experiments(range(i, i + 3), load_eval_rollouts=False) for i in exp_nums]
    titles = ['Double\nQ-learning', '5-step Double\nQ-learning', 'Our\napproach']
    colors = ['k', 'm', 'g']

    f_cumreward, axes_cumreward = plt.subplots(1, 1, figsize=(6, 3), sharey=False, sharex=True)
    if not hasattr(axes_cumreward, '__len__'):
        axes_cumreward = np.array([axes_cumreward])

    window = 32
    ylim = (0, 2100)
    xmax_timesteps = 8e5

    for exp, title, color in zip(all_exps, titles, colors):
        plot_cumreward(axes_cumreward[0], exp, window=window, ylim=ylim, plot_indiv=False, convert_to_time=True, label=title, color=color)

    # set same x-axis
    xmax = xmax_timesteps * (0.25 / 3600.)
    for ax in axes_cumreward:
        ax.set_ylim(ylim)
        ax.set_xlim((0, xmax))
        ax.yaxis.set_ticks(np.arange(0, 2100, 500))
        ax.set_yticklabels([''] * len(ax.get_yticklabels()))
        ax.yaxis.set_ticks_position('both')

        ax.legend(ncol=len(all_exps), loc='upper center', bbox_to_anchor=(0.5, 1.4))

    f_cumreward.text(0.5, -0.05, 'Time (hours)', ha='center', fontdict=font)

    ax = axes_cumreward[0]
    ax.set_yticklabels(['', '250', '500', '750', '1000', ''])
    ax.set_ylabel('Distance (m)', fontdict=font)

    # add y-axis on right side which is hallway lengths
    ax = axes_cumreward[-1]
    ax_twin = ax.twinx()
    ax_twin.set_xlim((ax.get_xlim()))
    ax_twin.set_ylim((ax.get_ylim()))
    ax_twin.set_yticks(ax.get_yticks())
    ax_twin.yaxis.tick_right()
    ax_twin.set_yticklabels(['', '3', '6', '9', '12', ''])
    ax_twin.set_ylabel('Hallway lengths', fontdict=font)
    ax_twin.yaxis.set_label_position("right")

    f_cumreward.savefig(os.path.join(SAVE_FOLDER, '{0}.png'.format(FILE_NAME)), bbox_inches='tight', dpi=200)
    plt.close(f_cumreward)

# plot_2445_2516()
# plot_2518_2577_and_2796_2819()
# plot_2578_2637_and_2820_2843()
# plot_2638_2697_and_2844_2867()
# plot_2698_2757_and_2868_2891()
# plot_2759_2794()
# plot_2893_3084()

# plot_empty_hallway_reset()
# plot_empty_hallway_lifelong()
# plot_cluttered_hallway_reset()
# plot_cluttered_hallway_lifelong()
#
# plot_priority_replay_empty_hallway_reset()
# plot_priority_replay_empty_hallway_lifelong()
# plot_priority_replay_cluttered_hallway_reset()
# plot_priority_replay_cluttered_hallway_lifelong()
#
# plot_design_decisions_empty_hallway_reset()
# plot_design_decisions_empty_hallway_lifelong()
# plot_design_decisions_cluttered_hallway_reset()
# plot_design_decisions_cluttered_hallway_lifelong()

# plot_dd_heatmap_empty_hallway_lifelong()
# plot_dd_heatmap_cluttered_hallway_lifelong()

plot_dd_cluttered_hallway_lifelong_outputs_loss()
plot_dd_cluttered_hallway_lifelong_horizon()
plot_dd_cluttered_hallway_lifelong_bootstrapping()

plot_cluttered_hallway_lifelong_standalone()