
from cnnClassifier.constants import *
from cnnClassifier.utils import read_yaml, create_directories
from cnnClassifier.entity.config_entity import DataIngestionConfig

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
        secret = self.secret.aws_credential

        create_directories([config.root_dir])
        data_ingestion_config = DataIngestionConfig(
            root_dir = config.root_dir,
            s3_bucket = secret.s3_bucket,
            s3_key = secret.s3_key,
            s3_secret_key= secret.s3_secret_key,
            object_key = secret.object_key,
            local_data_file=config.local_data_file

        )

        return data_ingestion_config