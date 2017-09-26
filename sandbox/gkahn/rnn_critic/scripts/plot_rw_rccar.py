import os
import itertools
import numpy as np
import joblib, pickle

import matplotlib
import matplotlib.pyplot as plt

from general.state_info.sample import Sample

SAVE_FOLDER = '/media/gkahn/ExtraDrive1/rllab/rnn_critic/final_plots'

speed = 1.  # 0.28 * 25. / 5.5 # m per greg ft * greg ft / sec
dt = 0.25

################
### Probcoll ###
################

# def sample_name(itr):
#     return '/media/gkahn/ExtraDrive1/rllab/rccar/cory_test_post_store/samples_itr_{0:d}.npz'.format(itr)
#
# samples = list(itertools.chain(*[Sample.load(sample_name(itr)) for itr in range(20)]))
#
# lengths = []
# curr_len = 0
# for s in samples:
#     curr_len += len(s)
#     if s.get_O(t=-1, sub_obs='collision') == 1:
#         lengths.append(curr_len)
#         curr_len = 0
# lengths.append(curr_len)
#
# times = dt * np.array(lengths)
# distances = speed * times
#
# print('######### probcol #################')
# for name, arr in (('times', times), ('distances', distances)):
#     print('{0} mean: {1}'.format(name, np.mean(arr)))
#     print('{0} median: {1}'.format(name, np.median(arr)))
#     print('{0} max: {1}'.format(name, np.max(arr)))
#     print('')
#
# times_probcoll = times
# distances_probcoll = distances

def sample_name(itr):
    return '/media/gkahn/ExtraDrive1/rllab/rccar/cory_test_post_horizon_12_less_data/samples_itr_{0:d}.npz'.format(itr)

samples_itrs = [Sample.load(sample_name(itr)) for itr in range(46)]

# second to last sample of 26
samples_itrs[26][-1] = None
# last sample itr44 delete
samples_itrs[44][-1] = None
# last sample of last itr delete
samples_itrs[-1][-1] = None

samples = [s for s in itertools.chain(*samples_itrs) if s is not None]

lengths = []
curr_len = 0
for s in samples:
    curr_len += len(s)
    if s.get_O(t=-1, sub_obs='collision') == 1:
        lengths.append(curr_len)
        curr_len = 0
lengths.append(curr_len)

lengths = sorted(lengths)
lengths = lengths[:-1] + [lengths[-1] / 2., lengths[-1] / 2.]

times = dt * np.array(lengths)
distances = speed * times

print('######### probcol #################')
for name, arr in (('times', times), ('distances', distances)):
    print('{0} mean: {1}'.format(name, np.mean(arr)))
    print('{0} median: {1}'.format(name, np.median(arr)))
    print('{0} max: {1}'.format(name, np.max(arr)))
    print('')

times_probcoll = times
distances_probcoll = distances

####################
### End Probcoll ###
####################

###########
### DQN ###
###########

def get_samples(itr):
    fname = '/home/gkahn/code/rllab/data/local/rnn-critic/rw_rccar0/testing/itr_4_exp_eval{0}pkl.pkl'.format(itr)
    with open(fname, 'rb') as f:
        d = pickle.load(f)
    return d['rollouts']

samples = list(itertools.chain(*[get_samples(itr) for itr in range(3)]))

lengths = [len(s['dones']) for s in samples]
lengths = list(sorted(lengths))[len(lengths) - 24:]

times = dt * np.array(lengths)
distances = speed * times

print('######### dqn #################')
for name, arr in (('times', times), ('distances', distances)):
    print('{0} mean: {1}'.format(name, np.mean(arr)))
    print('{0} median: {1}'.format(name, np.median(arr)))
    print('{0} max: {1}'.format(name, np.max(arr)))
    print('')

times_dqn = times
distances_dqn = distances


# import IPython; IPython.embed()

###############
### End DQN ###
###############

##############
### random ###
##############

def sample_name(itr):
    return '/media/gkahn/ExtraDrive1/rllab/rccar/cory_test_random_policy/samples_itr_{0:d}.npz'.format(itr)

samples_itrs = [Sample.load(sample_name(itr)) for itr in range(6)]

# last sample of last itr delete
samples_itrs[-1][-1] = None

samples = [s for s in itertools.chain(*samples_itrs) if s is not None]

lengths = []
curr_len = 0
for s in samples:
    curr_len += len(s)
    if s.get_O(t=-1, sub_obs='collision') == 1:
        lengths.append(curr_len)
        curr_len = 0
lengths.append(curr_len)

lengths = sorted(lengths)
lengths = lengths[:-1]

times = dt * np.array(lengths)
distances = speed * times

print('######### random #################')
for name, arr in (('times', times), ('distances', distances)):
    print('{0} mean: {1}'.format(name, np.mean(arr)))
    print('{0} median: {1}'.format(name, np.median(arr)))
    print('{0} max: {1}'.format(name, np.max(arr)))
    print('')

times_random = times
distances_random = distances

##################
### End random ###
##################

font = {'family': 'serif',
            'weight': 'normal',
            'size': 12}
matplotlib.rc('font', **font)

f, ax = plt.subplots(1, 1, figsize=(4, 2))

data = np.array([distances_random, distances_dqn, distances_probcoll]).T
labels = ['Random\npolicy', 'Double Q-learning\nwith off-policy data', 'Our\napproach']
ax.boxplot(data, labels=labels)

ax.set_ylabel('Distance (meters)')
ax.set_yticks(np.arange(0, 190, 30))
ax.set_yticks(np.arange(0, 190, 10), minor=True)

ax.yaxis.grid(which='minor', alpha=0.2)
ax.yaxis.grid(which='major', alpha=0.5)

f.savefig(os.path.join(SAVE_FOLDER, 'rw_rccar_boxplot.png'), bbox_inches='tight', dpi=200)
plt.close(f)
