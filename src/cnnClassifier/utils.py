import os
import sys
import yaml

import numpy as np
import pandas as pd
from cnnClassifier.logger import logging
from cnnClassifier.exception import CustomException
import pickle
from keras.models import save_model, load_model

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


@ensure_annotations
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
def pixel_to_image(data, img_height:int, img_width:int):
    """
    convert a series of string numbers to a 3D matrix as imege format

    Args:
        data (Dataframe): data to be used for converting
        img_height (int): image height
        img_width (int): image width              
    """
    
    # Three channels with same pixels
    images = np.empty((len(data), img_height, img_width, 3))
    for i, pixel_string in enumerate(data):
        temp = [float(pixel) for pixel in pixel_string.split(' ')]
        temp = np.asarray(temp).reshape(img_height, img_width)
        for j in range(3):
            images[i,:,:,j] = temp
    images/= 255.0        

    return images

#@ensure_annotations
def save_object(path: Path, obj:Any, h5=False):
    """
    save data as pickel or h5 

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