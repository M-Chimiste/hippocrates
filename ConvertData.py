# ConvertData.py
# Purpose: To convert histological sample images to numpy array test
# and train files.

# -----Begin Imports-------
import pandas as pd
import numpy as np
import cv2
import os
import shutil
import random
import time
from tqdm import tqdm
# -----End Imports------

TRAIN_DIR = 'train'
TEST_DIR = 'test'
TRAIN_DEST_DIR = 'converted_train_no_rot'
TEST_DEST_DIR = 'converted_test'
VAL_DIR = 'validation_no_rotation'
FILES_AT_ONCE = 60000

training_key = pd.read_csv('train_labels.csv')


# Take a filename, remove the file extention and return just the name
def ConvertFilename(filename):
    name = filename.split('.')[0]
    return name


# Function: ConvertData
# Purpose: Takes the image files and converts them into a numpy array with the training
# labels.
def ConvertData(directory, destination, train=True):
    data = []
    image_rotations = [0, 90, 180, 270]
    for image in tqdm(os.listdir(directory)):
        path = os.path.join(directory, image)
        name = ConvertFilename(image)
        if train:
            for rotation in image_rotations:
                label = int(training_key.loc[training_key['id'] == name]['label'])
                img = cv2.imread(path, cv2.IMREAD_COLOR)
                (cols, rows) = img.shape[:2]
                M = cv2.getRotationMatrix2D((cols/2, rows/2), rotation, 1)
                rotatated_image = cv2.warpAffine(img, M, (cols, rows))
                data = [[np.array(rotatated_image)], [np.array(label)]]
                np.save(f'{destination}/{name}-{rotation}.npy', data)
        else:
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            data = [np.array(img)]
            np.save(f'{destination}/{name}.npy', data)


# Function: CreateValidationSet
# Purpose: To take a directory of converted data, shuffle the contents
# and create a cross validation set in a new directory.
def CreateValidationSet(source_directory, dest_directory, pct=0.2):
    all_files = os.listdir(source_directory)
    total_files = len(all_files)
    random.shuffle(all_files)
    number_validation = int(total_files*pct) - 1
    validation_set = all_files[0:number_validation]
    for a_file in tqdm(validation_set):
        current_file_path = os.path.join(source_directory, a_file)
        new_file_path = os.path.join(dest_directory, a_file)
        shutil.move(current_file_path, new_file_path)


# Function: BalanceData
# Purpose: Input a list of data and then ensure that the data is balanced
def BalanceData(data):
    lengths = [len(data[0]), len(data[1])]
    min_length = min(lengths)
    zeros = data[0]
    ones = data[1]
    zeros = zeros[:min_length]
    ones = ones[:min_length]
    data = zeros + ones
    random.shuffle(data)
    return data


# CreateBalancedBlockedData
# Purpose: To create balanced blocks of data to load into a CNN
def CreateBalancedBlockedData(source_directory, dest_directory, block_size=FILES_AT_ONCE):
    training_list = os.listdir(source_directory)
    random.shuffle(training_list)

    current = 0
    increment = block_size
    not_maximum = True
    maximum = len(training_list)

    while not_maximum:
        
        # empty dictionary to hold image and choice data
        choices = {
            0: [],
            1: []
        }
        # iterate over the data through a specified increment
        print(f"Iterating through {current}-{current+increment} of {maximum}")
        for img in tqdm(training_list[current:current+increment]):
            full_path = os.path.join(source_directory, img)
            data_file = np.load(full_path)
            data_file = list(data_file)
            choice = int(data_file[1])
            if choice == 0:
                converted_choice = [1, 0]
                choices[choice].append([data_file[0], converted_choice])
            elif choice == 1:
                converted_choice = [0, 1]
                choices[choice].append([data_file[0], converted_choice])
        
        print("BalancingData")
        shuffled_data = BalanceData(choices)
        choices = None  # Get the data out of memory

        print("Reshaping Data")
        unique_id = int(time.time())
        # define and create the training and testing sets
        x_train = np.array([i[0][0] for i in shuffled_data]).reshape(-1, 96, 96, 3)
        y_train = np.array([i[1] for i in shuffled_data])
        np.save(f"{dest_directory}/x_train_{unique_id}.npy", x_train)
        np.save(f"{dest_directory}/y_train_{unique_id}.npy", y_train)
        current += increment
        if current >= maximum:
            not_maximum = False

# Operations

training_list = os.listdir(TRAIN_DIR)
testing_list = os.listdir(TEST_DIR)

CreateValidationSet(TRAIN_DEST_DIR, VAL_DIR)

ConvertData(TRAIN_DIR, TRAIN_DEST_DIR)
ConvertData(TEST_DIR, TEST_DEST_DIR, train=False)

CreateBalancedBlockedData("converted_train_rotated", "block_data_train_rot")
CreateBalancedBlockedData("validation_rotation", "block_data_test_rot", block_size=15000)
