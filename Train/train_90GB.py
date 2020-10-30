import sys
import os
import tensorflow as tf
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense, Activation, Reshape
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras import callbacks
from keras.layers import LeakyReLU
import time
from PIL import Image
import warnings
import pandas as pd
from keras.initializers import glorot_uniform
from keras.regularizers import l2
from skimage import exposure
import skimage

start = time.time()

train_data = "D:\\Dataset\\train"
train_label = "D:\\Dataset\\train.csv"
validation_data_path = "C:\\Users\\Kishan\\Desktop\\Diabetes\\data\\validation"
df_train=pd.read_csv(train_label)

df_train["image"]=df_train["image"].map('{}.jpeg'.format)
df_train["level"]=df_train["level"].map('class{}'.format)

"""
Parameters
"""
run_opts = tf.RunOptions(report_tensor_allocations_upon_oom = True)

#set parameters from training
img_width, img_height = 256, 256
batch_size = 32

#dynamic parameter for model
nb_filters1 = 32
nb_filters2 = 64
nb_filters3 = 128

conv1_size = 5
conv2_size = 3
conv3_size = 3

pool_size = 2
classes_num = 5
lr = 1e-3
epochs=25
channel=1

#For CLAHE Pre Processsing Function
def AHE(img):
    img = exposure.rescale_intensity(img,out_range=(0,1))
    img_adapteq = exposure.equalize_adapthist(img1, clip_limit=0.03)
    return img_adapteq

#Creating Sequential model  
model=Sequential()
#1st Input Layer
model.add(Conv2D(filters=nb_filters1, kernel_size=(conv1_size, conv1_size), padding ="valid", input_shape=(img_width, img_height, channel)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(pool_size, pool_size)))

#Hidden Layer
model.add(Conv2D(filters=nb_filters2, kernel_size=(conv2_size, conv2_size), padding ="valid"))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(pool_size, pool_size)))

#Hidden layer
model.add(Conv2D(filters=nb_filters3, kernel_size=(conv3_size, conv3_size), padding ="valid"))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(pool_size, pool_size)))

#Convert 2D array into 1-D array
model.add(Flatten())
model.add(Dense(500, activation='relu'))
model.add(Dropout(0.5))

#Dropout layer for protecting model from overfitting
model.add(Dense(500, activation='relu'))
model.add(Dropout(0.5))

#Output Layer
model.add(Dense(classes_num, activation='softmax'))

model.summary()

#Set a parameter for a training
model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.Adam(lr=lr),
              metrics=['accuracy'])

#Preprocess traning dataset 
train_datagen = ImageDataGenerator(
    rescale=1./255,
    fill_mode='nearest'
    preprocessing_function=AHE
    )

#preprocess Testing dataset
test_datagen = ImageDataGenerator(
    rescale=1./255,
    fill_mode='nearest'
    preprocessing_function=AHE
    )


#As per dataset label=folder name. so get a data from folders
train_generator=train_datagen.flow_from_dataframe(dataframe=df_train,
                                            directory=train_data,
                                            x_col="image",
                                            y_col="level",
                                            class_mode="categorical",
                                            target_size=(img_height, img_width),
                                            batch_size=batch_size,
                                            color_mode='grayscale',
                                            shuffle=True)

#validation data as per training data
validation_generator = test_datagen.flow_from_directory(
    validation_data_path,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    color_mode='grayscale',
    shuffle=True)


"""
Tensorboard log
To see the training graph
Run(cmd): tensorboard --logdir="path of log" 
"""
log_dir = './tf-log/'
tb_cb = callbacks.TensorBoard(log_dir=log_dir, histogram_freq=0,write_images=True,write_graph=True,update_freq='epoch',batch_size=batch_size)
sLog = time.strftime("%Y%m%d-%H%M", time.gmtime())
model_dir='./checkpoint'
os.makedirs(model_dir, exist_ok=True)
checkpoint_cb=keras.callbacks.ModelCheckpoint(filepath=model_dir + "/" + sLog + "model.h5", verbose=1,save_best_only=True, save_weights_only=False)
cbks = [tb_cb,checkpoint_cb]

#Start training
model.fit_generator(
    train_generator,
    epochs=epochs,
    validation_data=validation_generator,
    callbacks=cbks)

#save model
target_dir = './models/'
if not os.path.exists(target_dir):
  os.mkdir(target_dir)
model.save('./models/model.h5')
model.save_weights('./models/weights.h5')

#Calculate execution time
end = time.time()
dur = end-start

if dur<60:
    print("Execution Time:",dur,"seconds")
elif dur>60 and dur<3600:
    dur=dur/60
    print("Execution Time:",dur,"minutes")
else:
    dur=dur/(60*60)
    print("Execution Time:",dur,"hours")
