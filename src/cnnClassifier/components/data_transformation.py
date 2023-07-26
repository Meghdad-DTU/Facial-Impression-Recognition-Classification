
import sys
import os
import pandas as pd
from cnnClassifier.logger import logging
from cnnClassifier.exception import CustomException
from cnnClassifier.utils import pixel_to_matrix, save_object

from keras.utils import to_categorical
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
            pixel_column = 'pixels'
            target_column_name = 'emotion'             

            logging.info(f"Applying preprocessing object on train, validation and test dataframes")
            input_train_arr = preprocessing_obj.fit_transform(train_df[pixel_column])
            for i in range(len(input_train_arr)):
                path_file = os.path.join(self.config.local_train_dir, str(i)+'.png')
                save_img(path_file, input_train_arr[i])
            logging.info(f"Train set is saved as .png")         
            
            input_val_arr = preprocessing_obj.fit_transform(val_df[pixel_column])  
            for i in range(len(input_val_arr)):
                path_file = os.path.join(self.config.local_val_dir, str(i)+'.png')
                save_img(path_file, input_val_arr[i])         
            logging.info(f"Validation set is saved as .png")
            
            input_test_arr = preprocessing_obj.transform(test_df[pixel_column]) 
            for i in range(len(input_test_arr)):
                path_file = os.path.join(self.config.local_test_dir, str(i)+'.png')
                save_img(path_file, input_test_arr[i])  
            logging.info(f"Test set is saved as .png")
            

            logging.info(f"Changing target variable format to match with keras")
            target_train_arr = to_categorical(train_df[target_column_name], 7)   
            target_val_arr = to_categorical(val_df[target_column_name], 7)  
            target_test_arr = to_categorical(test_df[target_column_name], 7) 
            
            logging.info(f"Saving target variables as pickel file")
            target_variable = {'train': target_train_arr, 'validation': target_val_arr, 'test': target_test_arr }           
            save_object(path=self.config.local_target_file, obj=target_variable)
            
            logging.info('Saved preprocessing object')
            save_object(path=self.config.local_preprocessor_file, obj=preprocessing_obj)