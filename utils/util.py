import json
import os
from collections import OrderedDict
from itertools import repeat
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms

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
    current_path = os.getcwd()
    for j, counter in enumerate(list_counters):
        count = len(counter)
        counter_sorted = dict(sorted(counter.items()))
        # import pdb; pdb.set_trace()
        plt.bar(range(count), list(counter_sorted.values()), align='center')
        plt.xticks(range(count), list(counter_sorted.keys()))
        plt.savefig(current_path / 'images' / f'images/{split}_{_FACTORS_IN_ORDER[j]}.jpg')
        plt.close()

def add_margin(
        img_list: torch.Tensor,
        labels: torch.Tensor,
        predictions: torch.Tensor,
        margins: int,
        idx2label: dict,
        font: Path,
):
    transformToPil = transforms.Compose([transforms.ToPILImage()])
    transformToTensor = transforms.Compose([transforms.ToTensor()])
    new_images = []
    for k, img in enumerate(img_list):
        actual_predicted_label = np.argmax(list(predictions[k]))
        if labels[k] == actual_predicted_label:
            color = "green"
        else:
            color = "red"
        pil_img = transformToPil(img.cpu())
        width, height = pil_img.size
        bottom = top = right = left = margins
        new_width = width + right + left
        new_height = height + top + bottom
        result = Image.new(pil_img.mode, (new_width, new_height), color)
        result.paste(pil_img, (left, top))

        font = ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf", 12)
        add_text = ImageDraw.Draw(result)
        rgb_mean = list(map(int, [np.mean(np.asarray(result)[:, :, 0]),
                                  np.mean(np.asarray(result)[:, :, 1]),
                                  np.mean(np.asarray(result)[:, :, 2])]))
        rgb_mean = [(el + 150) % 255 for el in rgb_mean]
        add_text.text((10, 10), text=f"l:{idx2label[labels[k].item()]}",
                       font=font, fill=(rgb_mean[0], rgb_mean[1], rgb_mean[2]), align="center")
        add_text.text((10, 30), text=f"p:{idx2label[float(actual_predicted_label)]}",
                      font=font, fill=(rgb_mean[0], rgb_mean[1], rgb_mean[2]), align="center")
        result = transformToTensor(result)
        new_images.append(result)
    return new_images


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
