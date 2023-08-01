import sys
from cnnClassifier.exception import CustomException
from cnnClassifier.logger import logging
from cnnClassifier.config.configuration import configurationManeger
from cnnClassifier.components.validation import Evaluation


STAGE_NAME = "Model Validation Stage"

class ModelValidationPipeline:
    def __init__(self):
        pass

    def main(self):
        config = configurationManeger()
        val_config = config.get_validation_config()
        evaluation = Evaluation(val_config)
        evaluation.evaluation()
        evaluation.save_score()
        

if __name__ == '__main__':
    try:        
        logging.info(f'>>>>>>> stage {STAGE_NAME} started <<<<<<<<')
        obj = ModelValidationPipeline()
        obj.main()
        logging.info(f'>>>>>>> stage {STAGE_NAME} completed <<<<<<<<')
    
    except Exception as e:
        raise CustomException(e, sys)