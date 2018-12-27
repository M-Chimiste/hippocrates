# Evaluation.py
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
Y_NAME = 'y_train'
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
    return text_prediction,prediction



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
        
        positive = []
        negative = []
        false_pos = []
        false_neg = []

        act_pos = []
        act_neg = []

        for i in range(len(testing_file_x)):
            data = testing_file_x[i]
            result = testing_file_y[i]
            
            if result == 0:
                act_neg.append(result)
                cancer = False
            elif result == 1:
                act_pos.append(result)
                cancer = True
            
            text_prediction, prediction = MakePrediction(model, data)
            numeric_prediction = np.argmax(prediction)  # 0 or 1

            if result == prediction and cancer:
                positive.append(result)
            elif result == prediction and cancer is False:
                negative.append(result)
            elif result != prediction and cancer:
                false_neg.append(numeric_prediction)
            elif result != prediction and cancer is False:
                false_pos.append(numeric_prediction)
        
        total_cancer_samples = len(positive)
        total_non_cancer_samples = len(negative)
        total_samples = len(testing_file_x)

        acc_pos = round(float(len(positive) / total_cancer_samples), 3)
        acc_neg = round(float(len(negative) / total_non_cancer_samples), 3)
        overall = round(float((len(positive) + len(negative)) / total_samples), 3)
        acc_fpos = round(float(len(false_pos) / total_cancer_samples), 3)
        acc_fneg = round(float(len(false_neg) / total_non_cancer_samples), 3)

        metrics = {"Overall": overall,
                "Positive Accuracy": acc_pos,
                "Negative Accuracy": acc_neg,
                "False Positive Rate": acc_fpos,
                "False Negative Rate": acc_fneg
        }

        confusion = {"Positive": len(positive),
                    "Negative": len(negative),
                    "False Pos": len(false_pos),
                    "False Neg": len(false_neg)}
        return metrics, confusion
            

# CreatePredictionsFromImages TODO
# Will input a directory of images and output a report detailing the
# algorithm's predictions as an excel file
def CreatePredictionsFromImages(input_images_dir):
    return


# CreatePredictionsFromReport TODO
# Will imput a filename for a report that already exists (xlsx format).
# Then will take the sample names, and the image dir and create a prediction outputting an 
# excel file
def CreatePredictionsFromReport(report_file, image_dir):
    return

