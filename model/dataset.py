import itertools
import os
import pickle
from collections import Counter
from pathlib import Path

import h5py
import numpy as np
from sklearn.preprocessing import LabelEncoder
from torchvision import transforms
from torchvision.datasets.vision import VisionDataset

# FILE_NAME = '3dshapes.h5'
# NAME_LABELS = 'name_labels.pkl'
# FILE_NAME = 'first10.hdf5'
# NAME_LABELS = 'first10_labels.pkl'
# FILE_NAME = 'first20.hdf5'
# NAME_LABELS = 'first20_labels.pkl'
FILE_NAME = 'first10per.hdf5'
NAME_LABELS = 'first10per_labels.pkl'
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

def get_categorical_labels_list_retrieval(root: Path):
    """ Function returns a numerical encoding for the
    name labels"""
    with open(os.path.join(root, NAME_LABELS), 'rb') as f:
        labels = pickle.load(f)
    list_unique_words = list(Counter(list(itertools.chain.from_iterable(labels))).keys())
    list_new_labels = []
    idx2label = []
    label2idx = []
    le = LabelEncoder()
    le.fit(list_unique_words)
    label2idx.append(dict(zip(le.classes_, le.transform(le.classes_))))
    idx2label.append(dict(zip(le.transform(le.classes_), le.classes_)))
    list_new_labels = []
    for label in labels:
        list_new_labels.append(le.transform(label))
        # list_new_labels.append((le.transform(label)).astype(np.float))
    # with open(root / 'first10_name_labels_categorical.pkl', 'wb') as f:
    #     pickle.dump(list_new_labels, f)
    # print(list_new_labels)
    return np.array(list_new_labels), label2idx, idx2label
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
        if train is True:
            img_train = []
            labels_train = []
            no_len = int(1 * train_len)
            for index in idxs[0:no_len]:
                img_train.append(images[index])
                labels_train.append(list_new_labels[index])
            # import pdb; pdb.set_trace()
            self.images = np.array(img_train).reshape([no_len, 64, 64, 3])
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


class Shapes3dRetrieval(VisionDataset):
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
        print(f"Loading images from {FILE_NAME}")
        images = dataset['images'][:]
        print("Finished loading images")
        idxs = np.array(range(0, total_len))
        np.random.shuffle(idxs)
        ############ retrieval labels and idx <-> label dictionaries
        list_new_labels_ret,\
             label2idx_ret, idx2label_ret = get_categorical_labels_list_retrieval(root)
        self.label2idx_ret = label2idx_ret
        self.idx2label_ret = idx2label_ret
        ############ initial labels and idx <-> label dictionaries
        print("Loading labels")
        labels_init = dataset['labels'][:]
        list_new_labels_init,\
            label2idx_init, idx2label_init = get_categorical_labels_list(labels_init)
        self.label2idx_init = label2idx_init
        self.idx2label_init = idx2label_init
        print("labels transformed")
        if train is True:
            img_train = []
            # labels_train_actual = []
            labels_train_init = []
            labels_train_ret = []
            no_len = int(1 * train_len)
            for index in idxs[0:no_len]:
                img_train.append(images[index])
                labels_train_ret.append(list_new_labels_ret[index])
                labels_train_init.append(list_new_labels_init[index])
                # labels_train_actual.append(labels_init[index])
            # import pdb; pdb.set_trace()
            self.images = np.array(img_train).reshape([no_len, 64, 64, 3])
            self.labels_new_ret = labels_train_ret
            self.labels_new_init = labels_train_init
            # self.labels = labels_train_actual
        else:
            img_test = []
            labels_test_ret = []
            labels_test_init = []
            # labels_test_actual = []
            for index in idxs[train_len:]:
                img_test.append(images[index])
                labels_test_ret.append(list_new_labels_ret[index])
                labels_test_init.append(list_new_labels_init[index])
                # labels_test_actual.append(labels_init[index])
            self.images = np.array(img_test).reshape([test_len, 64, 64, 3])
            self.labels_new_ret = labels_test_ret
            self.labels_new_init = labels_test_init
            # self.labels = labels_test_actual

    def __getitem__(self, index):
        label_ret = self.labels_new_ret[index]
        label_init = self.labels_new_init[index]
        # labels = self.labels[index]
        img = self.images[index]
        if self.transform is not None:
            img = self.transform(img)
        return img, label_ret, label_init
    def __len__(self):
        return len(self.labels_new_ret)
