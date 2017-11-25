from keras import backend as K
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Lambda, Conv2D
from keras.layers import Input, Dropout
from keras.layers.convolutional import MaxPooling2D
from keras.models import Model, load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam, RMSprop, SGD
import numpy as np
import urllib
import keras
import tensorflow as tf
import horovod.keras as hvd

hvd.init()
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.visible_device_list = str(hvd.local_rank())
K.set_session(tf.Session(config=config))

datapath = '/WORK/hpjeonGIT/data/'
# Ref: http://agnesmustar.com/2017/05/25/build-vgg16-scratch-part-ii/
target_size=(224, 224)
batch_size=32

def get_batches(directory, target_size=target_size, batch_size=batch_size, shuffle=False):
    datagen = ImageDataGenerator()
    return datagen.flow_from_directory(directory=directory,
                                          target_size=target_size,
                                          batch_size=batch_size,
                                          class_mode='categorical',
                                          shuffle=shuffle)
batches = get_batches(path+'train', shuffle=True)
valid_batches = get_batches(path+'valid', batch_size=batch_size*2, shuffle=False)

json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
model.load_weights('model_first.h5')

opt = SGD(lr = 0.001)
opt = hvd.DistributedOptimizer(opt)
callbacks = [
    hvd.callbacks.BroadcastGlobalVariablesCallback(0),
    hvd.callbacks.MetricAverageCallback(),
    hvd.callbacks.LearningRateWarmupCallback(warmup_epochs=2, verbose=1),
    keras.callbacks.ReduceLROnPlateau(patience=3, verbose=1),
]

model.compile(optimizer=opt,
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

model.fit_generator(batches, steps_per_epoch=batches.samples//batch_size // hvd.size(), nb_epoch=50,
                validation_data=valid_batches, validation_steps=valid_batches.samples//batch_size//hvd.size())


if hvd.rank() == 0:
  model.save_weights("model_second.h5")
  print("Saved mode in the second step")

