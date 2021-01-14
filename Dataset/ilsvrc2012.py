import os
import glob
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import matplotlib.pyplot as plt
from scipy.io import loadmat
from torch.utils.data import Dataset

dataset_path = '/home/agaurav/Documents/Datasets/ILSVRC2012/'


def get_train_df():
    mat = loadmat(dataset_path + 'ILSVRC2012_devkit_t12/data/meta.mat')

    dict_wnid_to_label = dict()
    dict_label_to_wnid = dict()
    dict_image_label = {'image': [], 'label': []}

    # print(mat['synsets'].dtype)
    for element in mat['synsets']:
        label = element[0]['ILSVRC2012_ID'][0][0] - 1
        wnid = element[0]['WNID'][0]

        dict_wnid_to_label[wnid] = label
        dict_label_to_wnid[label] = wnid

    test_dir = dataset_path + 'ILSVRC2012_img_train/'
    for category in os.listdir(test_dir):
        cat_dir = test_dir + category
        for img_path in glob.glob(cat_dir + '/*.JPEG'):
            dict_image_label['image'].append(img_path)
            dict_image_label['label'].append(dict_wnid_to_label[category])

    return pd.DataFrame(dict_image_label)


class ILSVRC2012Dataset(Dataset):

    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        image = torch.from_numpy(plt.imread(self.df['image'].values[idx]).astype('float32'))
        if len(image.shape) == 2:
            image = image.repeat(3, 1, 1)
        elif len(image.shape) == 3 and image.shape[2] == 3:
            image = image.permute(2, 0, 1)
        elif len(image.shape) == 3 and image.shape[2] == 4:
            image = image[:, :, 0:3]
            image = image.permute(2, 0, 1)
        else:
            print(image.shape, self.df['image'].values[idx])

        label = torch.tensor(self.df['label'].values[idx])
        if self.transform:
            image = self.transform(image)

        if image.shape != (3, 224, 224):
            print(image.shape, self.df['image'].values[idx])

        return image, label


def get_ilsvrc2012_train_dataset(transform=None):
    train_df = get_train_df()
    return ILSVRC2012Dataset(train_df, transform=transform)



