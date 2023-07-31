import sys
from cnnClassifier.exception import CustomException
from cnnClassifier.logger import logging
from cnnClassifier.config.configuration import configurationManeger
from cnnClassifier.components.prepare_callbacks import PrepareCallbacks
from cnnClassifier.components.training import Training

STAGE_NAME = "Model Training Stage"

class ModelTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = configurationManeger()
        prepare_callbacks_config = config.get_prepare_callbacks_config()
        prepare_callbacks = PrepareCallbacks(config=prepare_callbacks_config)
        callback_list = prepare_callbacks.get_tb_ckpt_es_callbacks()

        training_config = config.get_training_config()
        training = Training(config= training_config)
        training.get_base_model()
        training.train_valid_generator()
        training.train(callback_list= callback_list)
        

if __name__ == '__main__':
    try:        
        logging.info(f'>>>>>>> stage {STAGE_NAME} started <<<<<<<<')
        obj = ModelTrainingPipeline()
        obj.main()
        logging.info(f'>>>>>>> stage {STAGE_NAME} completed <<<<<<<<')
    
    except Exception as e:
        raise CustomException(e, sys)