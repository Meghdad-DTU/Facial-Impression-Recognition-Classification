
import sys
import os
import pandas as pd
from cnnClassifier.logger import logging
from cnnClassifier.exception import CustomException
from cnnClassifier.utils import pixel_to_matrix

from keras.preprocessing.image import save_img
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from box import ConfigBox

from cnnClassifier.config.configuration import DataTransformationConfig



class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config

    def get_data_transformer_object(self):
        try:            
            face_recognition_pipeline = Pipeline(
                steps=[
                    ('pixel_to_image', 
                    FunctionTransformer(pixel_to_matrix, kw_args={'img_height':48, 'img_width':48}))
                ]
            )
            return face_recognition_pipeline            
        
        except Exception as e:
            raise CustomException(e, sys)
    
    
    def initiate_data_transformation(self):
        if not os.path.exists(self.config.local_train_file):
            logging.info(f"WARNING: {self.config.local_train_file} does not exist!")
        
        elif not os.path.exists(self.config.local_val_file):
            logging.info(f"WARNING: {self.config.local_val_file} does not exist!")   

        elif not os.path.exists(self.config.local_test_file):
            logging.info(f"WARNING: {self.config.local_test_file} does not exist!")              
        
        else:  
            
            train_df = pd.read_csv(self.config.local_train_file)
            val_df = pd.read_csv(self.config.local_val_file)
            test_df = pd.read_csv(self.config.local_test_file)
            logging.info('Read train, validation and test data completed')

            logging.info("Obtaining preprocessing object")
            preprocessing_obj=self.get_data_transformer_object()    

            emotion_path = {0: self.config.local_train_angry_dir,
                            1: self.config.local_train_disgust_dir,
                            2: self.config.local_train_fear_dir,
                            3: self.config.local_train_happy_dir,
                            4: self.config.local_train_sad_dir,
                            5: self.config.local_train_surprise_dir,
                            6: self.config.local_train_nuetral_dir}                 

     
            logging.info(f"Applying preprocessing object on train, validation and test dataframes")
            emotions, images = preprocessing_obj.fit_transform(train_df)
            i=1
            for emotion, image in zip(emotions, images):
                path_config = emotion_path[emotion]
                path_file = os.path.join(path_config, str(i)+'.png')
                save_img(path_file, image)
                i+=1
            logging.info(f"Train set is saved as .png")  
            # update emotion_path for validation
            for emotion, path in emotion_path.items():
                path = path.replace('train', 'validation')
                emotion_path[emotion] = path       
            
            emotions, images = preprocessing_obj.fit_transform(val_df)  
            i=1
            for emotion, image in zip(emotions, images):
                path_config = emotion_path[emotion]
                path_file = os.path.join(path_config, str(i)+'.png')
                save_img(path_file, image)
                i+=1        
            logging.info(f"Validation set is saved as .png")
            # update emotion_path for test
            for emotion, path in emotion_path.items():
                path = path.replace('validation', 'test')
                emotion_path[emotion] = path
            emotions, images = preprocessing_obj.transform(test_df) 
            i=1
            for emotion, image in zip(emotions, images):
                path_config = emotion_path[emotion]
                path_file = os.path.join(path_config, str(i)+'.png')
                save_img(path_file, image)
                i+=1  
            logging.info(f"Test set is saved as .png")   