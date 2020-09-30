import argparse
import itertools
import os
import pickle
from collections import Counter
from pathlib import Path

import h5py
import numpy as np
from sklearn.preprocessing import LabelEncoder

# FILE_NAME = '3dshapes.h5'
# FILE_NAME = 'first10.hdf5'
FILE_NAME = 'first20.hdf5'

hue_dict = {0: "red", 0.1: "orange", 0.2: "yellow", 0.30000000000000004: "green",
            0.4: "light green", 0.5: "light blue", 0.6000000000000001: "blue",
            0.7000000000000001: "dark blue", 0.8: "purple", 0.9: "pink"}
scale_dict = {0.75: "extra tiny",
                0.8214285714285714: "super tiny",
                0.8928571428571428: "very tiny",
                0.9642857142857143: "tiny",
                1.0357142857142856: "small",
                1.1071428571428572: "big",
                1.1785714285714286: "very big",
                1.25: "huge"}
shape_dict = {0.0: "cubical",
                1.0: "cylindrical",
                2.0: "spherical",
                3.0: "ovoidal"}
orientation_dict = {-30: "minus thirty",
                    -25.714285714285715: "minus twenty-five",
                    -21.42857142857143: "minus twenty-one",
                    -17.142857142857142: "minus seventeen",
                    -12.857142857142858: "minus twelve",
                    -8.571428571428573: "minus eight",
                    -4.285714285714285: "minus four",
                    0.0: "zero",
                    4.285714285714285: "four",
                    8.57142857142857: "eight",
                    12.857142857142854: "twelve",
                    17.14285714285714: "seventeen",
                    21.42857142857143: "twenty-one",
                    25.714285714285715: "twenty-five",
                    30: "thirty"}
def get_labels_in_words_form(root: Path, FILE_NAME: str):
    """ Function takes in actual labels and creates a new pkl file
    containing words corresponding to the number labels based on the
    defined dictionaries above"""
    root = Path("/scratch/shared/beegfs/oncescu/coding/libs/pt/3d-shapes-template/data")
    dataset = h5py.File(os.path.join(root, FILE_NAME), 'r')
    labels = dataset['labels'][:]
    # images = dataset['images']
    dicts = [hue_dict, hue_dict, hue_dict, scale_dict, shape_dict, orientation_dict]
    
    labels_names = []
    for idx, label in enumerate(labels):
        label_name_list = []
        for i in range(0, 6):
            label_name_list.append(dicts[i][label[i]])
        labels_names.append(label_name_list)
    with open(root / 'first20_labels.pkl', 'wb') as f:
        pickle.dump(labels_names, f)

def get_categorical_labels_retrieval(root: Path,
                                     list_unique_words: list,
                                     labels_init: list):
    """ Function generates an "encoding" for each word and saves a new
    pkl file with these labels"""
    list_new_labels = []
    idx2label = []
    label2idx = []
    le = LabelEncoder()
    le.fit(list_unique_words)
    label2idx.append(dict(zip(le.classes_, le.transform(le.classes_))))
    idx2label.append(dict(zip(le.transform(le.classes_), le.classes_)))
    list_new_labels = []
    for label in labels_init:
        list_new_labels.append(le.transform(label))
    with open(root / 'first20_name_labels_categorical.pkl', 'wb') as f:
        pickle.dump(list_new_labels, f)
    return np.array(list_new_labels), label2idx, idx2label

def index_encoding(root: Path, labels_file_name: str):
    with open(root / labels_file_name, 'rb') as f:
        labels = pickle.load(f)
    list_unique_words = list(Counter(list(itertools.chain.from_iterable(labels))).keys())
    get_categorical_labels_retrieval(root, list_unique_words, labels)
            

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default="/scratch/shared/beegfs/oncescu/coding/libs/pt/3d-shapes-template/data",
    )
    args = parser.parse_args()
    index_encoding(args.root, 'first20_labels.pkl')
    # get_labels_in_words_form(args.root, FILE_NAME)

if __name__ == "__main__":
    main()
