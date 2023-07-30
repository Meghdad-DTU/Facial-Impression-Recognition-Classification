
import keras
from cnnClassifier.utils import save_object
from cnnClassifier.config.configuration import PrepareBaseModelConfig



class PrepareBaseModel:
    def __init__(self, config: PrepareBaseModelConfig):
        self.config = config

    def get_base_model(self):
        self.model = keras.applications.vgg16.VGG16(
            input_shape = self.config.params_image_size,
            weights = self.config.params_weights,
            include_top = self.config.params_include_top
        )

        save_object(path=self.config.model_path, obj= self.model, h5=True)

    @staticmethod
    def _prepare_full_model(model, classes, freeze_all, freeze_till, learning_rate):
        if freeze_all:
            for layer in model.layers:
                model.trainable = False
        elif (freeze_till is not None) and (freeze_till > 0):
            for layer in model.layers[:-freeze_till]:
                model.trainable = False
        
        stacked_model=keras.models.Sequential()
        stacked_model.add(model)
        stacked_model.add(keras.layers.Dropout(0.5))
        stacked_model.add(keras.layers.Flatten())
        stacked_model.add(keras.layers.BatchNormalization())
        stacked_model.add(keras.layers.Dense(32,kernel_initializer='he_uniform'))
        stacked_model.add(keras.layers.BatchNormalization())
        stacked_model.add(keras.layers.Activation('relu'))
        stacked_model.add(keras.layers.Dropout(0.5))
        stacked_model.add(keras.layers.Dense(32,kernel_initializer='he_uniform'))
        stacked_model.add(keras.layers.BatchNormalization())
        stacked_model.add(keras.layers.Activation('relu'))
        stacked_model.add(keras.layers.Dropout(0.5))
        stacked_model.add(keras.layers.Dense(32,kernel_initializer='he_uniform'))
        stacked_model.add(keras.layers.BatchNormalization())
        stacked_model.add(keras.layers.Activation('relu'))
        stacked_model.add(keras.layers.Dense(units=classes, activation='softmax'))
        
        full_model = stacked_model

        '''
        # without stacked model
        
        flatten_in = keras.layers.Flatten()(model.output)
        prediction = keras.layers.Dense(
            units=classes,
            activation='softmax'
        )(flatten_in)

        full_model = keras.models.Model(
            inputs=model.input,
            outputs=prediction
        )
        '''

        full_model.compile(
            optimizer=keras.optimizers.SGD(learning_rate=learning_rate),
            loss = keras.losses.CategoricalCrossentropy(),
            metrics=['accuracy']
                           )
        
        full_model.summary()
        return full_model
    
    def update_base_model(self):
        self.full_model = self._prepare_full_model(
            model = self.model,
            classes=self.config.params_classes,
            freeze_all=False,
            freeze_till=4,
            learning_rate = self.config.params_learning_rate
        )

        save_object(path=self.config.updated_model_path, obj=self.full_model, h5=True)

        

        