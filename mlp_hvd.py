from keras import backend as K
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Lambda, Conv2D
from keras.layers import GlobalAveragePooling2D, Input, Dropout
from keras.layers.convolutional import MaxPooling2D
from keras.models import Model, load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam, RMSprop, SGD
import numpy as np
import urllib
import keras
import tensorflow as tf
import horovod.keras as hvd
import sys
import json
import time
hvd.init()
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.visible_device_list = str(hvd.local_rank())
K.set_session(tf.Session(config=config))

datapath = '/tmp/CET/KERAS/'
target_size=(150,150)
batch_size=128

def preprocess_image(im):
    vgg_mean = np.array([123.68, 116.779, 103.939], dtype=np.float32)
    #im = cv2.resize(cv2.imread(path), (224, 224)).astype(np.float32)
    im = (im - vgg_mean)
    return im[:, ::-1] # RGB to BGR

def create_mlp(num_classes):
    # we initialize the model
    model = Sequential() 
    model.add(Dense(64, activation='relu', input_shape=(150,150,1)))
    model.add(Dropout(0.2))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(num_classes, activation='softmax'))
    return model

def create_cnn(num_classes):    
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=(150, 150, 1)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(16, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    return model

def get_batches(directory, target_size=target_size, batch_size=batch_size, 
                shuffle=False):
    datagen = ImageDataGenerator()
    return datagen.flow_from_directory(directory=directory,
                                       target_size=target_size,
                                       color_mode = 'grayscale',
                                       batch_size=batch_size,
                                       class_mode='categorical',
                                       shuffle=shuffle)


batches = get_batches(datapath+'train', shuffle=True)
valid_batches = get_batches(datapath+'valid', batch_size=batch_size*2, 
                            shuffle=False)


if hvd.rank() == 0:
    with open("labels.txt", "w") as labels:
        labels.write(json.dumps(batches.class_indices))

num_classes = len(batches.class_indices)
model = create_cnn(num_classes)

opt = keras.optimizers.Adadelta(1.0 * hvd.size())
#opt = SGD(lr=0.01*hvd.size())
#opt = Adam(lr=0.1)
opt = hvd.DistributedOptimizer(opt)

callbacks = [
    hvd.callbacks.BroadcastGlobalVariablesCallback(0),
    hvd.callbacks.MetricAverageCallback(),            
    hvd.callbacks.LearningRateWarmupCallback(warmup_epochs=5, verbose=1),
    keras.callbacks.ReduceLROnPlateau(patience=10, verbose=1),
]


model.compile(optimizer=opt,
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

t0 = time.time()
fit = model.fit_generator(batches, 
                          steps_per_epoch=batches.samples//batch_size//hvd.size(), 
                          callbacks=callbacks,
                          nb_epoch=200,
                          validation_data=valid_batches, 
                          validation_steps=
                          valid_batches.samples//batch_size//hvd.size())

print("fitting time =", time.time() - t0, " at ", hvd.rank())
# serialize model to JSON
# ref: https://machinelearningmastery.com/save-load-keras-deep-learning-models/
# serialize weights to HDF5
if hvd.rank() == 0:
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    model.save_weights("model_w1st.h5")
    print("Saved model to disk")

if (hvd.rank() == hvd.size()-1):    
    import pandas as pd
    fname = 'epoch_first.dat'
    pd.DataFrame(fit.history).to_csv(fname,float_format='%.3f', sep=' ')

