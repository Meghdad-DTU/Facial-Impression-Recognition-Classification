import sys
from cnnClassifier.exception import CustomException
from cnnClassifier.logger import logging
from cnnClassifier.config.configuration import configurationManeger
from cnnClassifier.components.prediction import Prediction

STAGE_NAME = "Prediction Stage"

class PredictionPipeline:
    def __init__(self, filename):
        self.filename = filename

    def main(self):
        config = configurationManeger()
        pred_config = config.get_prediction_config()
        prediction = Prediction(self.filename, pred_config)
        prediction.predict()
        

if __name__ == '__main__':
    try:        
        logging.info(f'>>>>>>> stage {STAGE_NAME} started <<<<<<<<')
        obj = PredictionPipeline()
        obj.main()
        logging.info(f'>>>>>>> stage {STAGE_NAME} completed <<<<<<<<')
    
    except Exception as e:
        raise CustomException(e, sys)