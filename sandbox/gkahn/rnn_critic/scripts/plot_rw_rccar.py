import os
import itertools
import numpy as np

import matplotlib
import matplotlib.pyplot as plt

from general.state_info.sample import Sample

SAVE_FOLDER = '/media/gkahn/ExtraDrive1/rllab/rnn_critic/final_plots'

################
### Probcoll ###
################

def sample_name(itr):
    return '/media/gkahn/ExtraDrive1/rllab/rccar/cory_test_post_store/samples_itr_{0:d}.npz'.format(itr)

samples = list(itertools.chain(*[Sample.load(sample_name(itr)) for itr in range(20)]))

lengths = []
curr_len = 0
for s in samples:
    curr_len += len(s)
    if s.get_O(t=-1, sub_obs='collision') == 1:
        lengths.append(curr_len)
        curr_len = 0
lengths.append(curr_len)

times = 0.25 * np.array(lengths)
speed = 1.  # 0.28 * 25. / 5.5 # m per greg ft * greg ft / sec
distances = speed * times

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

font = {'family': 'serif',
            'weight': 'normal',
            'size': 15}
matplotlib.rc('font', **font)

f, ax = plt.subplots(1, 1, figsize=(4, 2))

data = np.array([distances_probcoll]).T
labels = ['Our approach']
ax.boxplot(data, labels=labels)

ax.set_ylabel('Distance (meters)')
ax.set_yticks(np.arange(0, 190, 30))
ax.set_yticks(np.arange(0, 190, 10), minor=True)

ax.yaxis.grid(which='minor', alpha=0.2)
ax.yaxis.grid(which='major', alpha=0.5)

f.savefig(os.path.join(SAVE_FOLDER, 'rw_rccar_boxplot.png'), bbox_inches='tight', dpi=200)
plt.close(f)
