import os
from pathlib import Path

import h5py
import numpy as np
from sklearn.preprocessing import LabelEncoder
from torchvision import transforms
from torchvision.datasets.vision import VisionDataset

# FILE_NAME = '3dshapes.h5'
FILE_NAME = 'first10.hdf5'
def get_categorical_labels_list(labels_init: list):
    list_new_labels = []
    no_tasks = len(labels_init[0])
    print("Transforming labels")
    for i in range(0, no_tasks):
        le = LabelEncoder()
        labels_init_i = labels_init[:, i]
        labels_new_i = le.fit_transform(labels_init_i)
        list_new_labels.append([labels_new_i])
    list_new_labels = np.concatenate(list_new_labels).T
    return(list_new_labels)

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
        print("Finished loading labels")
        
        list_new_labels = get_categorical_labels_list(labels_init)
        print("labels transformed")

        # import pdb; pdb.set_trace()
        if train is True:
            print("Loading images")
            self.images = dataset['images'][0:train_len]
            print("Loading labels")
            self.labels_init = labels_init[0:train_len]
            self.labels_new = list_new_labels[0:train_len]
            print("Done")
        else:
            self.images = dataset['images'][train_len:]
            self.labels_init = labels_init[train_len:]
            self.labels_new = list_new_labels[train_len:]
    def __getitem__(self, index):
        label = self.labels_new[index]
        img = self.images[index]
        if self.transform is not None:
            img = self.transform(img)
        return img, label
    def __len__(self):
        return len(self.labels_new)
