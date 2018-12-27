# TestingGrounds.py
# Purpose: To evaluate the different trained models
# ------Begin Imports------
import fnmatch
import pandas as pd
import numpy as np
from keras.models import load_model
import keras.backend.tensorflow_backend as backend
import tensorflow as tf
import os
from tqdm import tqdm
# ------End Imports -------

# -----Start Global Variables-------
TESTING_DIR = "block_data_test_rot"
X_NAME = 'x_train'
Y_NAME = 'y_name'
MODEL_NAME =''
# ------End Global Variables------

model = load_model(MODEL_NAME)


# MakePrediction
# Purpose: To take a loaded model, create a prediction,
# format the output and pass it into other functions.
def MakePrediction(model, data):
    prediction = model.predict(data)
    scalar_pred = int(np.argmax(prediction))
    if scalar_pred == 0:
        text_prediction = "Not Metastatic"
    elif scalar_pred == 1:
        text_prediction = "Metastatic"
    dict_prediction = {"Final Prediction": text_prediction,
                        "Non-Metastatic Confidence": prediction[0],
                        "Metastaic Confidence": prediction[1]}
    return dict_prediction



# Function GetListofData
# Purpose: Recusively grab files in a directory that match a naming convention
def GetListofData(directory, name):
    files = []
    for root, dirnames, filenames in os.walk(directory):
        for filename in fnmatch.filter(filenames, f'*{name}*'):
            files.append(os.path.join(root, filename))
    return files


# NaiveValidate
# Purpose: To take testing data and create predictions then
# evaluate the predictions
def NaiveValidate(testing_dir, model):
    X_data = GetListofData(testing_dir, X_NAME)
    y_data = GetListofData(testing_dir, Y_NAME)
    
    X_data = X_data.sort()
    y_data = y_data.sort()
    for i in range(len(X_data)):
        
        full_path_x = os.path.join(testing_dir, X_data[i])
        full_path_y = os.path.join(testing_dir, y_data[i])

        testing_file_x = np.load(full_path_x)
        testing_file_y = np.load(full_path_y)
        for i in range(len(testing_file_x)):
            data = testing_file_x[i]
            result = testing_file_y[i]
            prediction = MakePrediction(model, data)
            
