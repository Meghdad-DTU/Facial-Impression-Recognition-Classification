import sys
from cnnClassifier.exception import CustomException
from cnnClassifier.logger import logging
from cnnClassifier.config.configuration import configurationManeger
from cnnClassifier.components.prepare_callbacks import PrepareCallbacks


### NOT Necessary to have it as Pipeline : It will be called for training model ######
STAGE_NAME = "Prepare Callbacks Stage"

class PrepareCallbacksTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = configurationManeger()
        prepare_callbacks_config = config.get_prepare_callbacks_config()
        prepare_callbacks = PrepareCallbacks(config=prepare_callbacks_config)
        callback_list = prepare_callbacks.get_tb_ckpt_es_callbacks()
        return callback_list
        

if __name__ == '__main__':
    try:        
        logging.info(f'>>>>>>> stage {STAGE_NAME} started <<<<<<<<')
        obj = PrepareCallbacksTrainingPipeline()
        obj.main()
        logging.info(f'>>>>>>> stage {STAGE_NAME} completed <<<<<<<<')
    
    except Exception as e:
        raise CustomException(e, sys)