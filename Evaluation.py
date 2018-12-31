# Evaluation.py
# Purpose: To evaluate the different trained models
# ------Begin Imports------
import fnmatch
import pandas as pd
import numpy as np
from keras.models import load_model
import keras.backend.tensorflow_backend as backend
import cv2
import tensorflow as tf
import os
from tqdm import tqdm
# ------End Imports -------

# -----Start Global Variables-------
TESTING_DIR = "block_data_test_rot"
X_NAME = 'x_train'
Y_NAME = 'y_train'
MODEL_NAME ='HippocratesCNN-30-epochs-0.001-lr-STAGE3-SGD'
# ------End Global Variables------

model = load_model(MODEL_NAME)


# MakePrediction
# Purpose: To take a loaded model, create a prediction,
# format the output and pass it into other functions.
def MakePrediction(data, model=MODEL_NAME):
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
    files = os.listdir(directory)
    target = []
    for i in files:
        if name in i:
            target.append(i)
    return target


# NaiveValidate
# Purpose: To take testing data and create predictions then
# evaluate the predictions
def NaiveValidate(testing_dir, training_key, model):
    image_list = os.listdir(testing_dir)
    full_paths = []
    file_names = []
    labels = {}
    positive = []
    negative = []
    false_pos = []
    false_neg = []

    act_pos = []
    act_neg = []
    
    for files in tqdm(image_list):
        path = os.path.join(testing_dir, files)
        full_paths.append(path)
        files = files.split('.')[0]
        file_names.append(files)
    
    for i in tqdm(range(len(full_paths))):
        image = full_paths[i]
        img = cv2.imread(image, cv2.IMREAD_COLOR)
        name = file_names[i]
       
        label = int(training_key.loc[training_key['id'] == name]['label'])
        
        
        data = np.array([img])
            
        if label == 0:
            act_neg.append(label)
            cancer = False
            
        elif label == 1:
            act_pos.append(label)
            cancer = True
            
        text_prediction, prediction = MakePrediction(data, model)
        numeric_prediction = np.argmax(prediction)  # 0 or 1

        if label == numeric_prediction and cancer:
            positive.append(label)
        elif label == numeric_prediction and cancer is False:
            negative.append(label)
        elif label != numeric_prediction and cancer:
            false_neg.append(numeric_prediction)
        elif label != numeric_prediction and cancer is False:
            false_pos.append(numeric_prediction)

    total_cancer_samples = len(positive)
    total_non_cancer_samples = len(negative)
    total_samples = len(full_paths)

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
def CreatePredictionsFromImages(input_images_dir, model=MODEL_NAME, file_ext='.img'):
    image_files = GetListofData(input_images_dir, file_ext)
    data_list = []
    
    for image in image_files:
        img = cv2.imread(image, cv2.IMREAD_COLOR)
        data_list.append([np.array([img]), image])
    
    counter = 0
    report = pd.DataFrame()
    header = [
        'Sample',
        'Text Classification',
        'Confidence'
    ]
    for data in data_list:
        text_prediction, prediction = MakePrediction(data[0], model)
        scalar_pred = np.argmax(prediction)
        prediction = prediction[scalar_pred]
        sample = data.split('\\')[1]
        sample.strip(file_ext)
        rpt_dict = {'Sample': sample, 'Text Classification': text_prediction, 'Confidence': prediction}
        counter += 1
        #TODO create a temporary dataframe to house each row of data then use concat to pull them all together
        #TODO once complete last line is to save as a CSV or XLSX or some other format.
    return


# CreatePredictionsFromReport TODO
# Will imput a filename for a report that already exists (xlsx format).
# Then will take the sample names, and the image dir and create a prediction outputting an 
# excel file
def CreatePredictionsFromReport(report_file, image_dir, file_ext= '.img'):
    return
