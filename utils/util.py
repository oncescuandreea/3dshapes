import json
from collections import OrderedDict
from itertools import repeat
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

_FACTORS_IN_ORDER = ['floor_hue', 'wall_hue', 'object_hue', 'scale', 'shape',
                     'orientation']

def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)

def read_json(fname):
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)

def write_json(content, fname):
    fname = Path(fname)
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)

def inf_loop(data_loader):
    ''' wrapper function for endless data loader. '''
    for loader in repeat(data_loader):
        yield from loader

def histogram_distribution(list_counters: list, split: str):
    """ function takes in list of counters for each batch and
        returns histogram for 1st epoch. Split can be train,
        validation, test"""
    for j, counter in enumerate(list_counters):
        count = len(counter)
        counter_sorted = dict(sorted(counter.items()))
        # import pdb; pdb.set_trace()
        plt.bar(range(count), list(counter_sorted.values()), align='center')
        plt.xticks(range(count), list(counter_sorted.keys()))
        plt.savefig(f'{split}_{_FACTORS_IN_ORDER[j]}.jpg')
        plt.close()

class MetricTracker:
    def __init__(self, *keys, writer=None):
        self.writer = writer
        # import pdb; pdb.set_trace()
        self._data = pd.DataFrame(index=keys, columns=['total', 'counts', 'average'])
        self.reset()
        
    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value, n=1):
        if self.writer is not None:
            self.writer.add_scalar(key, value)
        self._data.total[key] += value * n
        self._data.counts[key] += n
        self._data.average[key] = self._data.total[key] / self._data.counts[key]

    def avg(self, key):
        return self._data.average[key]
    
    def result(self):
        return dict(self._data.average)
