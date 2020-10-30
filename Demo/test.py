import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras.models import load_model
import time
from keras import optimizers
from keras.metrics import top_k_categorical_accuracy
from keras.applications.inception_v3 import preprocess_input

start = time.time()

def top_2_accuracy(in_gt, in_pred):
    return top_k_categorical_accuracy(in_gt, in_pred, k=2)

#Define Path
model_path = '../models/model.h5'
model_weights_path = '../models/weights.hdf5'

#Load the pre-trained models
model = load_model(model_path,custom_objects={'top_2_accuracy':top_2_accuracy})
model.load_weights(model_weights_path)

#Define image parameters
out_size = (512, 512)
img_width,img_height =512, 512
path="test"
#Define image parameters

#Prediction
def Prediction(test_path):
    datagen = ImageDataGenerator(
            out_size,
            preprocessing_function = preprocess_input)
    
    generator=datagen.flow_from_directory(path, color_mode="rgb", target_size=(img_width,img_height))
    array = model.predict_generator(generator)
    print(array)
    answer = np.argmax(array)
    if answer == 0:
        print("Predicted: Class 0")
        return "Class 0"
    elif answer == 1:
        print("Predicted: Class 1")
        return "Class 1"
    elif answer == 2:
        print("Predicted: Class 2")
        return "Class 2"
    elif answer == 3:
        print("Predicted: Class 3")
        return "Class 3"
    elif answer == 4:
        print("Predicted: Class 4")
        return "Class 4"
    else:
        print("Can't Predict")
