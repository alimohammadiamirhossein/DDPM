import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from collections import defaultdict
import copy
import random
from PIL import Image
import shutil
from urllib.request import urlretrieve
import os
import cv2
import time

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable

import torchvision
import torch.nn.functional as F
import albumentations as A
from albumentations.pytorch import ToTensorV2

import glob
from tqdm import tqdm

from torchvision.datasets import OxfordIIITPet
from torchvision.utils import make_grid
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split, ConcatDataset
from torchvision import transforms
import torchvision.transforms as tt

# Metrics
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix
import itertools
from sklearn.model_selection import train_test_split
from PIL import Image


def extract_archive(filepath):
    extract_dir = os.path.dirname(os.path.abspath(filepath))
    shutil.unpack_archive(filepath, extract_dir)


def merge_trainval_test(filepath):
    """
        #   Image CLASS-ID SPECIES BREED ID
        #   ID: 1:37 Class ids
        #   SPECIES: 1:Cat 2:Dog
        #   BREED ID: 1-25:Cat 1:12:Dog
        #   All images with 1st letter as captial are cat images
        #   images with small first letter are dog images
    """
    merge_dir = os.path.dirname(os.path.abspath(f'{filepath}/annotations/data.txt'))
    #if os.path.exists(merge_dir):
    #    print("Merged data is already exists on the disk. Skipping creating new data file.")
    #    return
    df = pd.read_csv(f"{filepath}/annotations/trainval.txt", sep=" ",
                     names=["Image", "ID", "SPECIES", "BREED ID"])
    df2 = pd.read_csv(f"{filepath}/annotations/test.txt", sep=" ",
                      names=["Image", "ID", "SPECIES", "BREED ID"])
    frame = [df, df2]
    df = pd.concat(frame)
    df.reset_index(drop=True)
    df.to_csv(f'{filepath}/annotations/data.txt', index=None, sep=' ')
    print("Merged data is created.")


def decode_label(label, map=None):
    if map is None:
        return decode_map[int(label)]
    else:
        return map[int(label)]


def preprocess_mask(mask):
    mask = np.float32(mask) / 255
    mask[mask == 2.0] = 0.0
    mask[(mask == 1.0) | (mask == 3.0)] = 1.0
    return mask


def display_image_grid(images_filenames, images_directory, masks_directory, predicted_masks=None):
    cols = 3 if predicted_masks else 2
    rows = len(images_filenames)
    figure, ax = plt.subplots(nrows=rows, ncols=cols, figsize=(10, 24))
    for i, image_filename in enumerate(images_filenames):
        image = Image.open(os.path.join(images_directory, f'{image_filename}.jpg')).convert('RGB')

        mask = Image.open(os.path.join(masks_directory, f'{image_filename}.png'))
        mask = preprocess_mask(mask)
        ax[i, 0].imshow(image)
        ax[i, 1].imshow(mask, interpolation="nearest")

        ax[i, 0].set_title("Image")
        ax[i, 1].set_title("Ground truth mask")

        ax[i, 0].set_axis_off()
        ax[i, 1].set_axis_off()

        if predicted_masks:
            predicted_mask = predicted_masks[i]
            ax[i, 2].imshow(predicted_mask, interpolation="nearest")
            ax[i, 2].set_title("Predicted mask")
            ax[i, 2].set_axis_off()
    plt.tight_layout()
    plt.show()
