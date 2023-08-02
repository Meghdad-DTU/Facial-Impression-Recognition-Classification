import pandas as pd
import keras
import matplotlib.pyplot as plt
from cnnClassifier.config.configuration import TrainingConfig
from cnnClassifier.utils import load_object, save_object, model_loss, confusion_matrix_display, pandas_classification_report
import keras

class Training:
    def __init__(self, config: TrainingConfig):
        self.config = config

    def get_base_model(self):
        self.model = load_object(path= self.config.updated_base_model_path, h5=True)

    def train_valid_generator(self):

        datagenerator_kwargs = dict(
            # Dividing the pixels by 255 for normalization  => range(0,1)
            # Scaling the pixels value in range(-1,1) by subtracting 0.5 and multiply 2
            rescale= ((1./255) - 0.5)*2,            
            # if there was no validation set:
            # validation_split = 0.20
        )

        dataflow_kwargs = dict(
            target_size=self.config.params_imgage_size[:-1],            
            batch_size = self.config.params_batch_size,
            interpolation= 'bilinear',
            # Important: if images have only one channel, color mode must be added
            color_mode="grayscale"

        )
        ## NOTE: Keras generator alway looks for subfolders (representing the classes). Images insight the subfolders are associated with a class.
        if self.config.params_is_augmentation:
            datagenerator = keras.preprocessing.image.ImageDataGenerator(
                featurewise_center=False,
                featurewise_std_normalization=False,
                rotation_range=10,
                width_shift_range=0.1,
                height_shift_range=0.1,
                zoom_range=.1,
                horizontal_flip=True,
                **datagenerator_kwargs
            )

        else:
            datagenerator = keras.preprocessing.image.ImageDataGenerator(
                **datagenerator_kwargs
            )
        ## NOTE: subset is added when we use validation_split, where directory for both training and validation is the same.
        ## NOTE: shuffle= False for validation as we want to check the performance model using predict_generator 
        self.valid_generator = datagenerator.flow_from_directory(
            directory= self.config.validation_data,
            #subset= "validation",
            shuffle= False,            
            **dataflow_kwargs
            )
        
        self.train_generator = datagenerator.flow_from_directory(
            directory= self.config.training_data,
            #subset= "training",
            shuffle= True,           
            **dataflow_kwargs
            )
        
    def train(self, callback_list: list):
        self.steps_per_epoch = self.train_generator.samples // self.train_generator.batch_size
        self.validation_steps = self.valid_generator.samples // self.valid_generator.batch_size

        history = self.model.fit(
            self.train_generator,
            validation_data = self.valid_generator,
            epochs= self.config.params_epochs,
            steps_per_epoch= self.steps_per_epoch,
            validation_steps=self.validation_steps,
            callbacks = callback_list
            )
        
        model_loss(history)
        filenames = self.valid_generator.filenames
        nb_samples = len(filenames)
        predict = self.model.predict_generator(self.valid_generator, steps = nb_samples)
        predict_classes = predict.argmax(axis=1)
        labels = ["angry", "disgust", "fear", 'happy', 'neutral', 'sad', 'surprise']
        report = pandas_classification_report(self.valid_generator.classes, predict_classes, labels)
        print(report)
        confusion_matrix_display(self.valid_generator.classes, predict_classes, labels)
        

        save_object(path= self.config.trained_model_path, obj=self.model, h5=True)  