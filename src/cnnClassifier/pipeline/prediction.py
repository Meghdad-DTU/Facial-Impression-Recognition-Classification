import sys
sys.path.append('/home/paladin/Downloads/Facial_Impression_Recognition_Calassification/src')


import os
import numpy as np
from cnnClassifier.exception import CustomException
from cnnClassifier.logger import logging
from cnnClassifier.config.configuration import PredictionConfig
import cv2
from cnnClassifier.utils import load_model



class PredictionPipeline:
    def __init__(self, filename, config: PredictionConfig):
        self.config = config
        self.filename = filename

    def predict(self):
        emotion_dict = {0: "Neutral", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Sad", 5: "Surprised", 6: "Neutral"}

        model = load_model(h5_path=self.config.path_of_model, json_path= self.config.path_of_model_json)              
        
        facecasc = cv2.CascadeClassifier(self.config.pre_trained_face_detector)
        image = cv2.imread(self.filename)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        #faces = facecasc.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=10)
        faces = facecasc.detectMultiScale(image, scaleFactor=1.2, minNeighbors=6)
        print("No of faces : ",len(faces))
        
        predict ={}
        i =1
        for (x, y, w, h) in faces:
            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 1)
            # If the input of trained model has one chanels
            roi_gray = gray[y:y + h, x:x + w] 
            # If the input of trained model has three chanels         
            #roi_color = image[y:y+h, x:x+w]               
            # Croping 
            cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)            
            cropped_img = ((cropped_img/255.) - 0.5)*2           
            
            prediction = model.predict(cropped_img)
            maxindex = int(np.argmax(prediction))  

            print("person ",i," : ",emotion_dict[maxindex]) 
            predict["person "+str(i)] = emotion_dict[maxindex]
                           
            cv2.putText(image, emotion_dict[maxindex], (x+10, y-20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            i+=1 
       
        ################################################################
        # To save predicted imeges
        #file_name = os.path.basename(self.filename )
        #dir_name = os.path.dirname(self.filename) 
        #cv2.imwrite(os.path.join(dir_name, 'pred_'+file_name), image)
        #################################################################       

        
        return  predict

