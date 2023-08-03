
from cnnClassifier.config.configuration import EvaluationConfig
from cnnClassifier.utils import load_model, save_json
import keras



class Evaluation:
    def __init__(self, config: EvaluationConfig):
        self.config = config
    
    def _test_generator(self):

        datagenerator_kwargs = dict(
            # Dividing the pixels by 255 for normalization  => range(0,1)
            # Scaling the pixels value in range(-1,1) by subtracting 0.5 and multiply 2
            rescale= ((1./255) - 0.5)*2           
            
            
        )

        dataflow_kwargs = dict(
            target_size=self.config.params_image_size[:-1],            
            batch_size = self.config.params_batch_size,
            interpolation= 'bilinear',
            color_mode="grayscale"
        )

        datagenerator = keras.preprocessing.image.ImageDataGenerator(
                **datagenerator_kwargs
        )
        
        ## NOTE: subset is added when we use validation_split, where directory for both training and validation is the same.
        ## NOTE: shuffle= False for test as we want to check the performance model using predict_generator 
        self.test_generator = datagenerator.flow_from_directory(
            directory= self.config.test_data,            
            shuffle= False,            
            **dataflow_kwargs
            )
       
         
    def evaluation(self):
        model = load_model(h5_path= self.config.path_of_model, json_path= self.config.path_of_model_json)    
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=self.config.all_params['LEARNING_RATE']),
            loss = keras.losses.CategoricalCrossentropy(),
            metrics=['accuracy']
                           )  
          
        self._test_generator()
        self.score = model.evaluate(self.test_generator)

    def save_score(self):
        scores = {'loss': self.score[0], 'accuracy': self.score[1]}
        save_json(path='scores.json', data=scores)

