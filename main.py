import sys
from cnnClassifier.logger import logging
from cnnClassifier.exception import CustomException
from cnnClassifier.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline

STAGE_NAME = "Data Ingestion Stage"

try:
    logging.info(f'>>>>>>> {STAGE_NAME} started <<<<<<<<')
    obj = DataIngestionTrainingPipeline()
    obj.main()
    logging.info(f'>>>>>>> {STAGE_NAME} completed <<<<<<<<')
    
except Exception as e:
    raise CustomException(e, sys)

import sys
from cnnClassifier.logger import logging
from cnnClassifier.exception import CustomException
from cnnClassifier.pipeline.stage_02_data_transformation import DataTransformationTrainingPipeline

STAGE_NAME = "Data Transformation Stage"

try:
    logging.info(f'>>>>>>> {STAGE_NAME} started <<<<<<<<')
    obj = DataTransformationTrainingPipeline()
    obj.main()
    logging.info(f'>>>>>>> {STAGE_NAME} completed <<<<<<<<')
    
except Exception as e:
    raise CustomException(e, sys)

import sys
from cnnClassifier.logger import logging
from cnnClassifier.exception import CustomException
from cnnClassifier.pipeline.stage_03_prepare_base_model import PrepareBaseModelTrainingPipeline

STAGE_NAME = "Prepare Base Model Stage"

try:
    logging.info(f'>>>>>>> {STAGE_NAME} started <<<<<<<<')
    obj = PrepareBaseModelTrainingPipeline()
    obj.main()
    logging.info(f'>>>>>>> {STAGE_NAME} completed <<<<<<<<')
    
except Exception as e:
    raise CustomException(e, sys)

import sys
from cnnClassifier.logger import logging
from cnnClassifier.exception import CustomException
from cnnClassifier.pipeline.stage_04_training import ModelTrainingPipeline

STAGE_NAME = "Model Training Stage"

try:
    logging.info(f'>>>>>>> stage {STAGE_NAME} started <<<<<<<<')
    obj = ModelTrainingPipeline()
    obj.main()
    logging.info(f'>>>>>>> stage {STAGE_NAME} completed <<<<<<<<')
    
except Exception as e:
    raise CustomException(e, sys)