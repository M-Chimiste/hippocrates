# ConvertData.py
# Purpose: To convert histological sample images to numpy array test and train files.

#-----Begin Imports-------
import pandas as pd
import numpy as np
import cv2
import os
from tqdm import tqdm
from random import shuffle

TRAIN_DIR = 'train'
TEST_DIR = 'test'
TRAIN_DEST_DIR = 'converted_train'
TEST_DEST_DIR = 'converted_test'

training_key = pd.read_csv('train_labels.csv')


# Take a filename, remove the file extention and return just the name
def ConvertFilename(filename):
    name = filename.split('.')[0]
    return name


def ConvertData(directory, destination, train=True):
    data = []
    for image in tqdm(os.listdir(directory)):
        path = os.path.join(directory,image)
        name = ConvertFilename(image)
        if train:
            label = int(training_key.loc[training_key['id'] == name]['label'])
            #label = label['label']
            #label = int(label)
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            data = [[np.array(img)], [np.array(label)]]
            np.save(f'{destination}/{name}.npy', data)
        else:
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            data = [np.array(img)]
            np.save(f'{destination}/{name}.npy', data)


training_list = os.listdir(TRAIN_DIR)
testing_list = os.listdir(TEST_DIR)


#ConvertData(TRAIN_DIR, TRAIN_DEST_DIR)
ConvertData(TEST_DIR,TEST_DEST_DIR, train=False)

