# Trainer.py
# Purpose: To train a neural network on histopathology slides

# -----Begin Imports-------
import os
import tensorflow as tf
import keras
import keras.backend.tensorflow_backend as backend
from keras.callbacks import TensorBoard
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, SpatialDropout2D
from keras.layers import Conv2D, SeparableConv2D, MaxPooling2D, Activation
from keras.optimizers import Adam, SGD
import numpy as np
import random
import time
import datetime
# ------End Imports-------

# ------Begin Global Constants------
USE_GPU = True
GPU_PERCENTAGE = 1.0
BATCH_SIZE = 64
TEST_SIZE = 100
EPOCHS = 1
LEARNING_RATE = 1e-3
DECAY = 1e-4
OPT = keras.optimizers.adam(lr=LEARNING_RATE, decay=DECAY)
TRAINING_DIR = 'converted_train'
FILES_AT_ONCE = 10000

# -------End Global Constants-------


# Function: LimitSession
# Purpose: Allow for the allocation of a percentage of GPU usage
# for the training.  This is to ensure that it is possible to limit
# GPU usage as necessary.
def LimitSession(gpu_fraction=0.75):
    gpu_options = tf.GPUOptions(per_process_gpu_memory=gpu_fraction)
    return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))


if USE_GPU:
    # Set the session to the desired GPU percentage
    backend.set_session(LimitSession(GPU_PERCENTAGE))


# ----Begin Model Construction-------
model = Sequential()

# Convolution Block 1
model.add(Conv2D(96, (7,7), padding='same',
                input_shape=(96,96,3),
                activation='relu'))
model.add(Conv2D(64, (3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))

# Convolution Block 2
model.add(SeparableConv2D(64, (7,7), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(SpatialDropout2D(0.3))

# Convolution Block 3
model.add(Conv2D(32, (1,1), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.1))

# Convolution Block 4
model.add(SeparableConv2D(32, (3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(SpatialDropout2D(0.1))

# Flatten layer
model.add(Flatten())

# Dense layer
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(2, activation='softmax'))

# Compile Model
model.compile(loss='sparse_categorical_crossentropy',
            optimizer=OPT,
            metrics=['accuracy'])

# Create a tensorboard object to allow for visualization of metrics
tensorboard = TensorBoard(log_dir=f"logs/Stage1-{LEARNING_RATE}-{EPOCHS}-{int(time.time())}")
#-----End Model Construction-----


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

  
training_list = os.listdir(TRAINING_DIR)
random.shuffle(training_list)

# Iterate over the total number of Epochs
for i in range(EPOCHS):
    current = 0
    increment = FILES_AT_ONCE
    not_maximum = True
    maximum = len(training_list)

    while not_maximum:
        try:
            # empty dictionary to hold image and choice data
            choices = {
                0: [],
                1: []
            }
            # iterate over the data through a specified increment
            for img in training_list[current:current+increment]:
                full_path = os.path.join(TRAINING_DIR, img)
                data_file = np.load(full_path)
                data_file = list(data_file)
                for info in data_file:
                    choice = info[1]
                    if choice == 0:
                        converted_choice = [1, 0]
                        choices[choice].append([info[0], converted_choice])
                    elif choice == 1:
                        converted_choice = [0, 1]
                        choices[choice].append([info[0], converted_choice])
            
            shuffled_data = BalanceData(choices)
            choices = None  # Get the data out of memory

            # define and create the training and testing sets
            x_train = np.array([i[0] for i in shuffled_data[:TEST_SIZE]]).reshape(-1, 96, 96, 3)
            y_train = np.array([i[1] for i in shuffled_data[:TEST_SIZE]])

            x_test = np.array([i[0] for i in shuffled_data[-TEST_SIZE:]]).reshape(-1, 96, 96, 3)
            y_test = np.array([i[1] for i in shuffled_data[-TEST_SIZE:]])

            # Begin training the model
            model.fit(x_train, y_train,
                    batch_size=BATCH_SIZE,
                    validation_data=(x_test, y_test),
                    shuffle=False,
                    verbose=1,
                    callbacks=[tensorboard])
            
            # Save the model per each iteration
            model.save(f'HippocratesCNN-{EPOCHS}-epochs-{LEARNING_RATE}-lr-STAGE1')
            current += increment

            if current >= maximum:
                not_maximum = False

        except Exception as e:
            print(e)
            pass # bite me
