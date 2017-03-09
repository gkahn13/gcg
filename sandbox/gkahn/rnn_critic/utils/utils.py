import time
from collections import defaultdict

##############
### Timing ###
##############

class TimeIt(object):
    def __init__(self, prefix=''):
        self.prefix = prefix
        self.start_times = dict()
        self.elapsed_times = defaultdict(int)

    def start(self, name):
        assert(name not in self.start_times)
        self.start_times[name] = time.time()

    def stop(self, name):
        assert(name in self.start_times)
        self.elapsed_times[name] += time.time() - self.start_times[name]
        self.start_times.pop(name)

    def elapsed(self, name):
        return self.elapsed_times[name]

    def reset(self):
        self.start_times = dict()
        self.elapsed_times = defaultdict(int)

    def __str__(self):
        s = ''
        names_elapsed = sorted(self.elapsed_times.items(), key=lambda x: x[1], reverse=True)
        for name, elapsed in names_elapsed:
            if 'total' not in self.elapsed_times:
                s += '{0}: {1: <10} {2:.1f}\n'.format(self.prefix, name, elapsed)
            else:
                assert(self.elapsed_times['total'] >= max(self.elapsed_times.values()))
                pct = 100. * elapsed / self.elapsed_times['total']
                s += '{0}: {1: <10} {2:.1f} ({3:.1f}%)\n'.format(self.prefix, name, elapsed, pct)
        return s

timeit = TimeIt()

##############
### Images ###
##############

def rgb2gray(rgb):
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    return 0.2989 * r + 0.5870 * g + 0.1140 * b

