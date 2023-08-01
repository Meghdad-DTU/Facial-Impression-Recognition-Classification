
from cnnClassifier.config.configuration import EvaluationConfig
from cnnClassifier.utils import load_object, save_json
import keras




class Evaluation:
    def __init__(self, config: EvaluationConfig):
        self.config = config
    
    def _valid_generator(self):

        datagenerator_kwargs = dict(
            # Dividing the pixels by 255 for normalization  => range(0,1)
            # Scaling the pixels value in range(-1,1) by subtracting 0.5 and multiply 2
            rescale= ((1./255) - 0.5)*2,            
            # if there was no validation set:
            # validation_split = 0.20
        )

        dataflow_kwargs = dict(
            target_size=self.config.params_image_size[:-1],            
            batch_size = self.config.params_batch_size,
            interpolation= 'bilinear'
        )

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
        
         
    def evaluation(self):
        model = load_object(path=self.config.path_of_model, h5=True)
        self._valid_generator()
        self.score = model.evaluate(self.valid_generator)

    def save_score(self):
        scores = {'loss': self.score[0], 'accuracy': self.score[1]}
        save_json(path='scores.json', data=scores)
