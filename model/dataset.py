import os
from pathlib import Path

import h5py
import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder

from skmultilearn.model_selection import iterative_train_test_split
from torchvision import transforms
from torchvision.datasets.vision import VisionDataset

FILE_NAME = '3dshapes.h5'
# FILE_NAME = 'first10.hdf5'
def get_categorical_labels_list(labels_init: list):
    list_new_labels = []
    idx2label = []
    label2idx = []
    no_tasks = len(labels_init[0])
    print("Transforming labels")
    for i in range(0, no_tasks):
        le = LabelEncoder()
        labels_init_i = labels_init[:, i]
        le.fit(labels_init_i)
        label2idx.append(dict(zip(le.classes_, le.transform(le.classes_))))
        idx2label.append(dict(zip(le.transform(le.classes_), le.classes_)))
        labels_new_i = le.transform(labels_init_i)
        list_new_labels.append([labels_new_i])
    list_new_labels = np.concatenate(list_new_labels).T
    return list_new_labels, label2idx, idx2label

class Shapes3d(VisionDataset):
    def __init__(self, root: Path, train: bool,
                 test_split: float,
                 transform: transforms.Compose = None):
        """
        root - Folder where file containing images and labels as arrays is stored
        """
        super().__init__(root, transform=transform)
        # import pdb; pdb.set_trace()
        dataset = h5py.File(os.path.join(root, FILE_NAME), 'r')
        total_len = len(dataset['labels'])
        test_len = int(test_split * total_len)
        train_len = total_len - test_len
        print("Loading labels")
        labels_init = dataset['labels'][:]
        print("Loading images")
        images = dataset['images'][:]
        print("Finished loading labels")
        idxs = np.array(range(0, total_len))
        np.random.shuffle(idxs)
        list_new_labels, label2idx, idx2label = get_categorical_labels_list(labels_init)
        self.label2idx = label2idx
        self.idx2label = idx2label
        print("labels transformed")
        # import pdb; pdb.set_trace()
        # idx_train, label_train,\
        #      idx_test, label_test = iterative_train_test_split(idxs,
        #                                                        list_new_labels,
        #                                                        test_size=test_split)
        # import pdb; pdb.set_trace()
        # if train is True:
        #     print("Loading images")
        #     self.images = dataset['images'][0:train_len]
        #     print("Loading labels")
        #     self.labels_init = labels_init[0:train_len]
        #     self.labels_new = list_new_labels[0:train_len]
        #     print("Done")
        # else:
        #     self.images = dataset['images'][train_len:]
        #     self.labels_init = labels_init[train_len:]
        #     self.labels_new = list_new_labels[train_len:]
        if train is True:
            img_train = []
            labels_train = []
            for index in idxs[0:train_len]:
                img_train.append(images[index])
                labels_train.append(list_new_labels[index])
            # import pdb; pdb.set_trace()
            self.images = np.array(img_train).reshape([train_len, 64, 64, 3])
            self.labels_new = labels_train
        else:
            img_test = []
            labels_test = []
            for index in idxs[train_len:]:
                img_test.append(images[index])
                labels_test.append(list_new_labels[index])
            self.images = np.array(img_test).reshape([test_len, 64, 64, 3])
            self.labels_new = labels_test
    def __getitem__(self, index):
        label = self.labels_new[index]
        img = self.images[index]
        if self.transform is not None:
            img = self.transform(img)
        return img, label
    def __len__(self):
        return len(self.labels_new)
