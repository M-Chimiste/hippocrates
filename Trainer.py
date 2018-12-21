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
from tqdm import tqdm
import fnmatch
# ------End Imports-------

# ------Begin Global Constants------
BATCH_SIZE = 128
TEST_SIZE = 100
EPOCHS = 5
LEARNING_RATE = 1e-3
DECAY = 1e-4
OPT = keras.optimizers.sgd(lr=LEARNING_RATE, momentum=0.001, decay=DECAY)
TRAINING_DIR = 'block_data_train_rot'
TESTING_DIR = 'block_data_test_rot'

# -------End Global Constants-------

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
model.compile(loss='categorical_crossentropy',
            optimizer=OPT,
            metrics=['accuracy'])

# Create a tensorboard object to allow for visualization of metrics
tensorboard = TensorBoard(log_dir=f"logs/Stage1-{LEARNING_RATE}-{EPOCHS}-{int(time.time())}")
#-----End Model Construction-----


# Function GetListofData
# Purpose: Recusively grab files in a directory that match a naming convention
def GetListofData(directory, name):
    files = []
    for root, dirnames, filenames in os.walk(directory):
        for filename in fnmatch.filter(filenames, f'*{name}*'):
            files.append(os.path.join(root, filename))
    return files


x_train_list = GetListofData(TRAINING_DIR, 'x_train')
y_train_list = GetListofData(TRAINING_DIR, 'y_train')
x_test_list = GetListofData(TESTING_DIR, 'x_train')
y_test_list = GetListofData(TESTING_DIR, 'y_train')

x_train_list.sort()
y_train_list.sort()
x_test_list.sort()
y_test_list.sort()

# Iterate over the total number of Epochs
for i in range(EPOCHS):
    idx = 0
    for n in tqdm(x_train_list):
        # Begin training the model
        x_train_file = x_train_list[idx]
        y_train_file = y_train_list[idx]
        x_test_file = x_test_list[idx]
        y_test_file = y_test_list[idx]


        x_train = np.load(x_train_file)
        y_train = np.load(y_train_file)
        x_test = np.load(x_test_file)
        y_test = np.load(y_test_file)
        model.fit(x_train, y_train,
                batch_size=BATCH_SIZE,
                validation_data=(x_test, y_test),
                shuffle=False,
                verbose=1,
                callbacks=[tensorboard])
        idx += 1    
        model.save(f'HippocratesCNN-{EPOCHS}-epochs-{LEARNING_RATE}-lr-STAGE1-SGD')

         

            
