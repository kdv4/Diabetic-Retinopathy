import sys
import os
import tensorflow as tf
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential, load_model
from keras.layers import Dropout, Flatten, Dense, Activation, Reshape
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras import callbacks
from keras.layers import LeakyReLU
import time
from PIL import Image
import warnings
import pandas as pd
start = time.time()

train_data = "C:\\Users\\Kishan\\Desktop\\Diabetes\\Dataset\\train"
train_label = "C:\\Users\\Kishan\\Desktop\\Diabetes\\Dataset\\train.csv"
validation_data_path = "C:\\Users\\Kishan\\Desktop\\Diabetes\\Dataset\\validation"
df_train=pd.read_csv(train_label)

df_train["image"]=df_train["image"].map('{}.jpeg'.format)
df_train["level"]=df_train["level"].map('class{}'.format)

"""
Parameters
"""
run_opts = tf.RunOptions(report_tensor_allocations_upon_oom = True)

#set parameters from training
img_width, img_height = 256, 256
batch_size = 16

#dynamic parameter for model
nb_filters1 = 32
nb_filters2 = 64
nb_filters3 = 128

conv1_size = 5
conv2_size = 3
conv3_size = 3

pool_size = 2
classes_num = 5
lr = 0.00001
epochs=30
channel=3

checkpoint = "C:\\Users\\Kishan\\Desktop\\Diabetes\\checkpoint\\20190402-0356model.h5"
model_path = '../models/model.h5'
model_weights_path = '../models/weights.h5'

#Load the pre-trained models
model = load_model(checkpoint)
#model=load_weight(model_weights_path)

model.summary()
sgd=optimizers.SGD(lr=lr, decay=1e-6, momentum=0.9, nesterov=True)

#Set a parameter for a training
model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.Adam(lr=lr),
              metrics=['accuracy'])

#Preprocess traning dataset
train_datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
    )

#preprocess Testing dataset
test_datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')


#As per dataset label=folder name. so get a data from folders
train_generator=train_datagen.flow_from_dataframe(dataframe=df_train,
                                            directory=train_data,
                                            x_col="image",
                                            y_col="level",
                                            class_mode="categorical",
                                            target_size=(img_height, img_width),
                                            batch_size=batch_size,
                                            color_mode='rgb')

#validation data as per training data
validation_generator = test_datagen.flow_from_directory(
    validation_data_path,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    color_mode='rgb')


"""
Tensorboard log
"""
log_dir = './tf-log/'
tb_cb = callbacks.TensorBoard(log_dir=log_dir, histogram_freq=0,write_images=True,write_graph=True,update_freq='batch',batch_size=batch_size)
sLog = time.strftime("%Y%m%d-%H%M", time.gmtime())
model_dir='./checkpoint'
os.makedirs(model_dir, exist_ok=True)
checkpoint_cb=keras.callbacks.ModelCheckpoint(filepath=model_dir + "/" + sLog + "model.h5", verbose=1,save_best_only=False, save_weights_only=False)

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
