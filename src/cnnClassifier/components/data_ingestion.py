
import os
import pandas as pd
from cnnClassifier.logger import logging
from io import StringIO
from pathlib import Path
import boto3
from cnnClassifier.utils import get_size
from cnnClassifier.entity.config_entity import DataIngestionConfig


class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config
        self.client = boto3.client('s3',
                      aws_access_key_id=self.config.s3_key,
                      aws_secret_access_key=self.config.s3_secret_key
                      )

    def dowload_file(self):
        if not os.path.exists(self.config.local_data_file):
            csv_obj = self.client.get_object(Bucket=self.config.s3_bucket, Key=self.config.object_key)
            body = csv_obj['Body']
            csv_string = body.read().decode('utf-8')
            df = pd.read_csv(StringIO(csv_string))
            df.to_csv(self.config.local_data_file, index=False, header=True)
            logging.info(f'{self.config.local_data_file} is downloaded!')

            logging.info('Train, validation and test split for data initiated')
            train_set = df[df.Usage == 'Training'].drop('Usage',axis=1)
            val_set = df[df.Usage == 'PublicTest'].drop('Usage',axis=1)
            test_set = df[df.Usage == 'PrivateTest'].drop('Usage',axis=1) 

            train_set.to_csv(self.config.local_train_file, index=False, header=True)
            val_set.to_csv(self.config.local_val_file, index=False, header=True)
            test_set.to_csv(self.config.local_test_file, index=False, header=True)

            logging.info("Train, validation and test split is done!")

        else:
            logging.info(f"File already exists of size : {get_size(Path(self.config.local_data_file))}")   