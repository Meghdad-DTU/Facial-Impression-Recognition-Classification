import os
import sys
import yaml

from cnnClassifier.logger import logging
from cnnClassifier.exception import CustomException

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
            logging.info('taml file {yaml_file} loaded successfully')
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
            logging.info(f'craeted directory at {path}')



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
