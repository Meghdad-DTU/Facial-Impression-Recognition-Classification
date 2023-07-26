
import sys
from cnnClassifier.logger import logging
from cnnClassifier.exception import CustomException

from cnnClassifier.config.configuration import configurationManeger
from cnnClassifier.components.data_transformation import DataTransformation



STAGE_NAME = "Data Transformation Stage"

class DataTransformationTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = configurationManeger()
        data_transformation_config = config.get_data_transformation_config()
        data_transformation = DataTransformation(config=data_transformation_config)
        data_transformation.initiate_data_transformation()
        


if __name__ == '__main__':
    try:        
        logging.info(f'>>>>>>> stage {STAGE_NAME} started <<<<<<<<')
        obj = DataTransformationTrainingPipeline()
        data = obj.main()
        logging.info(f'>>>>>>> stage {STAGE_NAME} completed <<<<<<<<')
    
    except Exception as e:
        raise CustomException(e, sys)