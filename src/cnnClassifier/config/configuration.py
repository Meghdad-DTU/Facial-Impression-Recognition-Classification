
import os
from pathlib import Path
from cnnClassifier.constants import *
from cnnClassifier.utils import read_yaml, create_directories
from cnnClassifier.entity.config_entity import (DataIngestionConfig, 
                                                DataTransformationConfig,
                                                PrepareBaseModelConfig, 
                                                PrepareCallbacksConfig,
                                                TrainingConfig,
                                                EvaluationConfig)

class configurationManeger:
    def __init__(self, 
                 config_filepath = CONFIG_FILE_PATH,
                 secret_filepath = SECRET_FILE_PATH,                 
                 params_filepath = PARAMS_FILE_PATH):
        
        self.config = read_yaml(config_filepath) 
        self.secret = read_yaml(secret_filepath)        
        self.params = read_yaml(params_filepath)

        create_directories([self.config.artifacts_root])

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion   
        
        create_directories([config.root_dir])
        
        data_ingestion_config = DataIngestionConfig(
            root_dir = config.root_dir,
            s3_bucket = self.secret.aws_credential.s3_bucket,
            s3_key = self.secret.aws_credential.s3_key,
            s3_secret_key = self.secret.aws_credential.s3_secret_key,
            object_key = self.secret.aws_credential.object_key,
            local_data_file = config.local_data_file,
            local_train_file = config.local_train_file,
            local_val_file = config.local_val_file,
            local_test_file = config.local_test_file

        )

        return data_ingestion_config
    
    def get_data_transformation_config(self) -> DataTransformationConfig:
        config = self.config.data_transformation             

        create_directories([config.root_dir,
                            config.local_train_angry_dir,
                            config.local_train_disgust_dir,
                            config.local_train_fear_dir,
                            config.local_train_happy_dir,
                            config.local_train_sad_dir,
                            config.local_train_surprise_dir,
                            config.local_train_nuetral_dir,
                            config.local_val_angry_dir,
                            config.local_val_disgust_dir,
                            config.local_val_fear_dir,
                            config.local_val_happy_dir,
                            config.local_val_sad_dir,
                            config.local_val_surprise_dir,
                            config.local_val_nuetral_dir,
                            config.local_test_angry_dir,
                            config.local_test_disgust_dir,
                            config.local_test_fear_dir,
                            config.local_test_happy_dir,
                            config.local_test_sad_dir,
                            config.local_test_surprise_dir,
                            config.local_test_nuetral_dir]
                            )

        data_trnsformation_config = DataTransformationConfig(
            root_dir = config.root_dir,
            local_train_angry_dir= config.local_train_angry_dir,
            local_train_disgust_dir= config.local_train_disgust_dir,
            local_train_fear_dir= config.local_train_fear_dir,
            local_train_happy_dir= config.local_train_happy_dir,
            local_train_sad_dir= config.local_train_sad_dir,
            local_train_surprise_dir= config.local_train_surprise_dir,
            local_train_nuetral_dir= config.local_train_nuetral_dir,
            local_val_angry_dir = config.local_val_angry_dir,
            local_val_disgust_dir= config.local_val_disgust_dir,
            local_val_fear_dir= config.local_val_fear_dir,
            local_val_happy_dir= config.local_val_happy_dir,
            local_val_sad_dir= config.local_val_sad_dir,
            local_val_surprise_dir= config.local_val_surprise_dir,
            local_val_nuetral_dir= config.local_val_nuetral_dir,
            local_test_angry_dir= config.local_test_angry_dir,
            local_test_disgust_dir= config.local_test_disgust_dir,
            local_test_fear_dir= config.local_test_fear_dir,
            local_test_happy_dir= config.local_test_happy_dir,
            local_test_sad_dir= config.local_test_sad_dir,
            local_test_surprise_dir= config.local_test_surprise_dir,
            local_test_nuetral_dir= config.local_test_nuetral_dir,           
            local_train_file = self.config.data_ingestion.local_train_file,
            local_val_file = self.config.data_ingestion.local_val_file,
            local_test_file = self.config.data_ingestion.local_test_file
        )

        return data_trnsformation_config
    
    def get_prepare_base_model_config(self) -> PrepareBaseModelConfig:
        config = self.config.prepare_base_model
        
        create_directories([config.root_dir])

        prepare_base_model_config = PrepareBaseModelConfig(
            root_dir = config.root_dir,           
            model_path = config.model_path,
            updated_model_path = config.updated_model_path,
            params_image_size = self.params.IMAGE_SIZE,
            params_learning_rate = self.params.LEARNING_RATE,
            params_include_top = self.params.INCLUDE_TOP,
            params_weights = self.params.WEIGHTS,
            params_classes = self.params.CLASSES

        )

        return prepare_base_model_config
    
    def get_prepare_callbacks_config(self) -> PrepareCallbacksConfig:
        config = self.config.prepare_callbacks
        model_ckpt_dir = os.path.dirname(config.ckeckpoint_model_filepath)

        create_directories([config.tensorboard_root_log_dir, model_ckpt_dir ])

        prepare_callbacks_config = PrepareCallbacksConfig(
           root_dir= config.root_dir,
           tensorboard_root_log_dir= config.tensorboard_root_log_dir,
           ckeckpoint_model_filepath=  config.ckeckpoint_model_filepath,
           patience = self.params.PATIENCE

        )

        return prepare_callbacks_config

    def get_training_config(self) -> TrainingConfig:
        config= self.config.training
        training_data_dir = os.path.dirname(self.config.data_transformation.local_train_angry_dir)
        validation_data_dir = os.path.dirname(self.config.data_transformation.local_val_angry_dir)
        
        create_directories([config.root_dir])

        training_config = TrainingConfig(
        root_dir= config.root_dir,
        trained_model_path= config.trained_model_path, 
        updated_base_model_path= self.config.prepare_base_model.updated_model_path, 
        training_data= training_data_dir,
        validation_data= validation_data_dir, 
        params_epochs= self.params.EPOCHS, 
        params_batch_size= self.params.BATCH_SIZE, 
        params_is_augmentation= self.params.AUGMENTATION,
        params_imgage_size= self.params.IMAGE_SIZE
        )

        return training_config
    
    def get_validation_config(self) -> EvaluationConfig:
        eval_config = EvaluationConfig(
            path_of_model= Path('artifacts/training/model.h5'),           
            validation_data= Path('artifacts/data_transformation/validation/'),
            all_params = self.params,
            params_image_size= self.params.IMAGE_SIZE,
            params_batch_size= self.params.BATCH_SIZE
            )
        
        return eval_config