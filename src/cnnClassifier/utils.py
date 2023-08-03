import os
import sys
import yaml

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, precision_recall_fscore_support

from cnnClassifier.logger import logging
from cnnClassifier.exception import CustomException
import pickle
from keras.models import save_model, load_model
from keras.models import model_from_json

import json
import joblib
from box import ConfigBox
from ensure import ensure_annotations
from pathlib import Path
from typing import Any
import base64

@ensure_annotations
def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """
    reads yaml file and returns
    Args:
        path to yaml (str): path like input
    Raises:
        valueError
    Returns:
        configBox: configBox type
    """

    try:
        with open(path_to_yaml) as yaml_file:
            content = yaml.safe_load(yaml_file)
            logging.info(f'yaml file {yaml_file.name} loaded successfully')
            return ConfigBox(content)
            
    except Exception as e:
        raise CustomException(e, sys)    

@ensure_annotations
def create_directories(path_to_directories: list, verbos=True):
    """
    create list of directories
    Args:
    path_to_directories(list): list of path of directories
    ignore_log(bool, optional): ignore if multiple dirs is to be created. Defaults to be False
    """
    for path in path_to_directories:
        os.makedirs(path, exist_ok=True)
        if verbos:
            logging.info(f'created directory at {path}')


#@ensure_annotations
def save_json(path:Path, data:dict):
    """
    save json data

    Args:
        path(Path): path to json file
        data (dict): data to be saved in json file        
    """

    with open(path, 'w') as f:
        json.dump(data, f, indent=4)

    logging.info(f"json file saved at {path}")


@ensure_annotations
def get_size(path: Path) -> str:
    """get size in KB

    Args:
        path (Path): path of the file

    Returns:
        str: size in KB
    """
    size_in_kb = round(os.path.getsize(path)/1024)
    return f"~ {size_in_kb} KB"

#@ensure_annotations
def pixel_to_matrix(df:pd.DataFrame, img_height:int, img_width:int, rgb=False):
    """
    convert a series of string numbers to a 3D matrix as imege format

    Args:
        df (Dataframe): data to be used for converting
        img_height (int): image height
        img_width (int): image width              
    """
    
    # Three channels with same pixels
    if rgb:
        channel = 3        
    else:
        channel = 1
        
    images = np.empty((len(df), img_height, img_width, channel))
    emotions = list()
    for ind, row in df.iterrows():
        temp = [float(pixel) for pixel in row['pixels'].split(' ')]
        temp = np.asarray(temp).reshape(img_height, img_width)
        for i in range(channel):
            images[ind,:,:,i] = temp

        emotions.append(row['emotion'])           

    return emotions, images

def save_model(h5_path: Path, json_path: Path, model: Any):
    """
    serialize model weights to HDF5

    Args:
        h5_path (Path): path to h5 file
        json_path (Path): path to json file
        model : model be saved in the file
         
    """
    model.save_weights(h5_path)
    model_json = model.to_json()
    with open(json_path, "w") as json_file:
        json_file.write(model_json)

    logging.info(f"model file saved at {h5_path}!")

def load_model(h5_path:Path, json_path:Path):
        #  loading the model in modular approach
        json_file = open(json_path, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        model = model_from_json(loaded_model_json)
        # load weights into new model
        model.load_weights(h5_path)        
        return model 

#@ensure_annotations
def save_object(path: Path, obj:Any, h5=False):
    """
    save object as pickel or h5 

    Args:
        path (Path): path to .pkl or .h5 file
        obj : object to be saved in the file
        h5 (boolean): if False, then the object is saved as .pkl      
    """
    try:
        dir_path = os.path.dirname(path)
        os.makedirs(dir_path, exist_ok=True)
        if h5:
            save_model(obj, path)
            logging.info(f"model is saved at {path}")
                
        else:
            with open(path, 'wb') as file_obj:
                pickle.dump(obj, file_obj)
            logging.info(f"pickel file saved at {path}")
    
    except Exception as e:
            raise CustomException(e, sys)
    
def load_object(path:Path, h5=False):
    """
    load object as pickel or h5 

    Args:
        path (Path): path to .pkl or .h5 file
        obj : object to be loaded 
        h5 (boolean): if False, then the object is loaded as .pkl      
    """
    try:
        if h5:
            return load_model(path, compile=True)
        else:
            with open(path, 'rb') as file_obj:
                return pickle.load(file_obj)
            
            logging.info(f"Object file loaded at {path}")
        
    except Exception as e:
        raise CustomException(e, sys)

def model_loss(history, label2='Validation Loss'):
    plt.figure(figsize=(8,4))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label=label2)
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epochs')
    plt.legend(loc='upper right')
    plt.grid(linestyle="--")
    plt.show();

def confusion_matrix_display(y_true, y_pred, classes):
    """
    This function prints and plots the confusion matrix
    Args:
        y_true: the actual value of y
        y_pred: the predicted valuye of y
        classes: list of label classes to be predicted
    """

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot(cmap=plt.cm.Blues)
    plt.show()

def pandas_classification_report(y_true, y_pred, classes):
    """
    This function returns scikit learn output metrics.classification_report 
    into CSV/tab-delimited format
    Args:
        y_true: the actual value of y
        y_pred: the predicted valuye of y
        classes: list of label classes to be predicted
    """
    metrics_summary = precision_recall_fscore_support(y_true=y_true, y_pred=y_pred)    
    avg = list(precision_recall_fscore_support(y_true=y_true, y_pred=y_pred, average='weighted'))
    metrics_sum_index = ['precision', 'recall', 'f1-score', 'support']
    class_report_df = pd.DataFrame(list(metrics_summary), index=metrics_sum_index)

    support = class_report_df.loc['support']
    total = support.sum() 
    avg[-1] = total

    class_report_df['avg / total'] = avg              
    df = class_report_df.T
    df.index = classes + ['avg / total']
    return  df