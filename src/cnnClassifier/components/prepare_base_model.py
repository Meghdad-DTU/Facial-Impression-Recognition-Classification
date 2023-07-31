
import keras
from cnnClassifier.utils import save_object
from cnnClassifier.config.configuration import PrepareBaseModelConfig



class PrepareBaseModel:
    def __init__(self, config: PrepareBaseModelConfig):
        self.config = config

    # In case, vgg16 is about to used as the base model
    def get_base_model(self):
        self.model = keras.applications.vgg16.VGG16(
            input_shape = self.config.params_image_size,
            weights = self.config.params_weights,
            include_top = self.config.params_include_top
        )

        save_object(path=self.config.model_path, obj= self.model, h5=True)

    @staticmethod
    def _prepare_full_model(base_model, classes, freeze_all, freeze_till, learning_rate, input_shape=None):
        if base_model is not None:        
            if freeze_all:
                for layer in base_model.layers:
                    base_model.trainable = False
            elif (freeze_till is not None) and (freeze_till > 0):
                for layer in base_model.layers[:-freeze_till]:
                    base_model.trainable = False
        
            model=keras.models.Sequential()
            model.add(base_model)
            model.add(keras.layers.Dropout(0.5))
            model.add(keras.layers.Flatten())
            model.add(keras.layers.BatchNormalization())
            model.add(keras.layers.Dense(32,kernel_initializer='he_uniform'))
            model.add(keras.layers.BatchNormalization())
            model.add(keras.layers.Activation('relu'))
            model.add(keras.layers.Dropout(0.5))
            model.add(keras.layers.Dense(32,kernel_initializer='he_uniform'))
            model.add(keras.layers.BatchNormalization())
            model.add(keras.layers.Activation('relu'))
            model.add(keras.layers.Dropout(0.5))
            model.add(keras.layers.Dense(32,kernel_initializer='he_uniform'))
            model.add(keras.layers.BatchNormalization())
            model.add(keras.layers.Activation('relu'))
            model.add(keras.layers.Dense(units=classes, activation='softmax'))

            '''
            # without stacked model
        
            flatten_in = keras.layers.Flatten()(base_model.output)
            prediction = keras.layers.Dense(
                units=classes,
                activation='softmax'
            )(flatten_in)

            full_model = keras.models.Model(
                inputs=base_model.input,
                outputs=prediction
            )
            '''
        
        else:
            assert input_shape is not None, " WARNING: Input shape mus be provided!"
            model = keras.models.Sequential()
            model.add(keras.layers.Convolution2D(filters=16, kernel_size=(7, 7), padding='same', name='image_array', input_shape= input_shape))
            model.add(keras.layers.BatchNormalization())
            model.add(keras.layers.Convolution2D(filters=16, kernel_size=(7, 7), padding='same'))
            model.add(keras.layers.BatchNormalization())
            model.add(keras.layers.Activation('relu'))
            model.add(keras.layers.AveragePooling2D(pool_size=(2, 2), padding='same'))
            model.add(keras.layers.Dropout(.5))

            model.add(keras.layers.Convolution2D(filters=32, kernel_size=(5, 5), padding='same'))
            model.add(keras.layers.BatchNormalization())
            model.add(keras.layers.Convolution2D(filters=32, kernel_size=(5, 5), padding='same'))
            model.add(keras.layers.BatchNormalization())
            model.add(keras.layers.Activation('relu'))
            model.add(keras.layers.AveragePooling2D(pool_size=(2, 2), padding='same'))
            model.add(keras.layers.Dropout(.5))

            model.add(keras.layers.Convolution2D(filters=64, kernel_size=(3, 3), padding='same'))
            model.add(keras.layers.BatchNormalization())
            model.add(keras.layers.Convolution2D(filters=64, kernel_size=(3, 3), padding='same'))
            model.add(keras.layers.BatchNormalization())
            model.add(keras.layers.Activation('relu'))
            model.add(keras.layers.AveragePooling2D(pool_size=(2, 2), padding='same'))
            model.add(keras.layers.Dropout(.5))

            model.add(keras.layers.Convolution2D(filters=128, kernel_size=(3, 3), padding='same'))
            model.add(keras.layers.BatchNormalization())
            model.add(keras.layers.Convolution2D(filters=128, kernel_size=(3, 3), padding='same'))
            model.add(keras.layers.BatchNormalization())
            model.add(keras.layers.Activation('relu'))
            model.add(keras.layers.AveragePooling2D(pool_size=(2, 2), padding='same'))
            model.add(keras.layers.Dropout(.5))

            model.add(keras.layers.Convolution2D(filters=256, kernel_size=(3, 3), padding='same'))

            model.add(keras.layers.BatchNormalization())
            model.add(keras.layers.Convolution2D(filters= classes, kernel_size=(3, 3), padding='same'))
            model.add(keras.layers.GlobalAveragePooling2D())
            model.add(keras.layers.Activation('softmax',name='predictions'))
        
        
        full_model = model            

        full_model.compile(
            optimizer=keras.optimizers.SGD(learning_rate=learning_rate),
            loss = keras.losses.CategoricalCrossentropy(),
            metrics=['accuracy']
                           )
        
        full_model.summary()
        return full_model
    
    def update_model(self):
        self.full_model = self._prepare_full_model(
            base_model = None, #self.model,
            classes=self.config.params_classes,
            freeze_all=False,
            freeze_till= None,
            learning_rate = self.config.params_learning_rate,
            input_shape = self.config.params_image_size

        )

        save_object(path=self.config.updated_model_path, obj=self.full_model, h5=True)

        

        