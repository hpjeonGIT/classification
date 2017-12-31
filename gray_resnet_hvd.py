from keras import backend as K
from keras.models import Sequential, Model
from keras.models import Model, load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam, RMSprop, SGD
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
from keras.layers import Dense, Activation, Flatten, Lambda, Conv2D


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

## main code is from:
## http://agnesmustar.com/2017/05/25/build-vgg16-scratch-part-ii/

datapath = '/work/jeonb/CET/KERAS/'
target_size=(224, 224)
batch_size=32

def get_batches(directory, target_size=target_size, batch_size=batch_size, 
                shuffle=False):
    datagen = ImageDataGenerator(rotation_range=45)
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
model = ResNet50(weights=None,input_shape=(224,224,1),classes=num_classes)

 


opt = keras.optimizers.Adadelta(1.0* hvd.size())
#opt = SGD(lr=0.01*hvd.size())
#opt = Adam(lr=0.1)
opt = hvd.DistributedOptimizer(opt)

callbacks = [
    hvd.callbacks.BroadcastGlobalVariablesCallback(0),
    hvd.callbacks.MetricAverageCallback(),            
    hvd.callbacks.LearningRateWarmupCallback(warmup_epochs=3, verbose=1),
    keras.callbacks.ReduceLROnPlateau(patience=5, verbose=1),
]


model.compile(optimizer=opt,
              loss='categorical_crossentropy', 
              metrics=['accuracy'])


t0 = time.time()
fit = model.fit_generator(batches, 
                    steps_per_epoch=batches.samples//batch_size//hvd.size(), 
                    callbacks=callbacks,
                    nb_epoch=100,
                    validation_data=valid_batches, 
                    validation_steps=
                    valid_batches.samples//batch_size//hvd.size())

print("fitting time =", time.time() - t0, " at ", hvd.rank())
# serialize model to JSON
# ref: https://machinelearningmastery.com/save-load-keras-deep-learning-models/
# serialize weights to HDF5
if hvd.rank() == 0:
    model_json = model.to_json()
    with open("resnet_model.json", "w") as json_file:
        json_file.write(model_json)
    model.save_weights("resnet_weight.h5")
    print("Saved model to disk")

if (hvd.rank() == hvd.size()-1):
    import pandas as pd
    fname = 'gray_epoch.dat'
    pd.DataFrame(fit.history).to_csv(fname,float_format='%.3f', sep=' ')

