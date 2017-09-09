import os, pickle
import numpy as np
import itertools

import matplotlib.pyplot as plt
from matplotlib import ticker
import matplotlib.cm as cm

from analyze_experiment import AnalyzeRNNCritic
from sandbox.gkahn.rnn_critic.utils.utils import DataAverageInterpolation

from sandbox.gkahn.rnn_critic.policies.mac_policy import MACPolicy
from robots.sim_rccar.analysis.analyze_sim_rccar import AnalyzeSimRCcar

EXP_FOLDER = '/media/gkahn/ExtraDrive1/rllab/s3/rnn-critic'
SAVE_FOLDER = '/media/gkahn/ExtraDrive1/rllab/rnn_critic/final_plots'

########################
### Load experiments ###
########################

def load_experiments(indices, create_new_envs=False):
    exps = []
    for i in indices:
        try:
            exps.append(AnalyzeRNNCritic(os.path.join(EXP_FOLDER, 'rccar{0:03d}'.format(i)),
                                         clear_obs=True,
                                         create_new_envs=create_new_envs,
                                         load_train_rollouts=False,
                                         load_eval_rollouts=True))
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
### Plot ###
############

def plot_cumreward(ax, analyze_group, color='k', label=None, window=20, success_cumreward=None, ylim=(10, 60)):
    data_interp = DataAverageInterpolation()
    if 'type' not in analyze_group[0].params['alg'] or analyze_group[0].params['alg']['type'] == 'interleave':
        min_step = max_step = None
        for i, analyze in enumerate(analyze_group):

            try:
                steps = np.array([r['steps'][0] for r in itertools.chain(*analyze.eval_rollouts_itrs)])
                values = np.array([np.sum(r['rewards']) for r in itertools.chain(*analyze.eval_rollouts_itrs)])

                steps, values = zip(*sorted(zip(steps, values), key=lambda k: k[0]))
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

    all_exps = [load_experiments(range(i, i + 3)) for i in range(2445, 2516, 3)]

    import IPython; IPython.embed()

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

    for i, xmax in enumerate([1e5, 2e5, 2e5, 4e5]):
        for ax in axes_cumreward[i, :]:
            ax.set_xlim((0, xmax))

    f_cumreward.savefig(os.path.join(SAVE_FOLDER, '{0}_cumreward.png'.format(FILE_NAME)), bbox_inches='tight', dpi=150)
    plt.close(f_cumreward)

def plot_2518_2577():
    FILE_NAME = 'rccar_2518_2577'

    all_exps = [load_experiments(range(i, i + 3)) for i in range(2518, 2577, 3)]

    import IPython; IPython.embed()

    f_cumreward, axes_cumreward = plt.subplots(5, 4, figsize=(12, 15), sharey=True, sharex=True)

    window = 8
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
# plot_1156_1200()
# plot_1202_1297()
# plot_1299_1319()
# plot_comparison_1319()
# plot_1321_1341()
# plot_1339_paths()
# plot_1343_1366()
# plot_1368_1382()
# plot_1384_1431()
# plot_1433_1492()

# plot_1494_1553()
# plot_1555_1650()

# plot_1652_1795()

# plot_1797_1904()
# plot_compare_1494_1905()
# plot_1912_2055()

# plot_2057_2128()
# plot_2130_2201()

# plot_2203_2346()

# plot_2348_2443()

# plot_compare()

plot_2445_2516()
# plot_2518_2577()