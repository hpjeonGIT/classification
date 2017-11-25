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
imoprt sys
import json

# ref: https://github.com/uber/horovod/blob/master/examples/keras_mnist_advanced.py
hvd.init()
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.visible_device_list = str(hvd.local_rank())
K.set_session(tf.Session(config=config))


datapath = '/WORK/hpjeonGIT/data/'
# Ref: http://agnesmustar.com/2017/05/25/build-vgg16-scratch-part-ii/
target_size=(224, 224)
batch_size=32

def preprocess_image(im):
    vgg_mean = np.array([123.68, 116.779, 103.939], dtype=np.float32)
    im = (im - vgg_mean)
    return im[:, ::-1] # RGB to BGR

def create_vgg16(x=None):
    # we initialize the model
    model = Sequential()
    # Conv Block 1
    model.add(Lambda(preprocess_image, input_shape=(224,224,3), output_shape=(224,224,3)))
    model.add(Conv2D(64, (3, 3), input_shape=(224,224,3), activation='relu', padding='same'))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    # Conv Block 2
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    # Conv Block 3
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    # Conv Block 4
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    # Conv Block 5
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    # FC layers
    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dense(4096, activation='relu'))
    model.add(Dense(1000, activation='softmax'))
    return model

def get_batches(directory, target_size=target_size, batch_size=batch_size, shuffle=False):
    datagen = ImageDataGenerator()
    return datagen.flow_from_directory(directory=directory,
                                          target_size=target_size,
                                          batch_size=batch_size,
                                          class_mode='categorical',
                                          shuffle=shuffle)

def get_weights(path, download=False):
    urllib.urlretrieve ("https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels.h5", model_path)

batches = get_batches(path+'train', shuffle=True)
valid_batches = get_batches(path+'valid', batch_size=batch_size*2, shuffle=False)

initial_model = create_vgg16()
initial_model.load_weights(model_path)

#### 
if hvd.rank() == 0:
  with open('labels.txt','w') as labels:
    labels.write(json.dumps(batches.class_indices))
   
initial_model = create_vgg16()
#initial_model.load_weights(model_path) # we may begin from scratch

x = Dense(batches.num_class, activation='softmax')(initial_model.layers[-2].output)
model = Model(initial_model.input, x)
# for layer in initial_model.layers: layer.trainable=False # for scratch build
#opt = Adam(lr=0.001)
opt = SGD(lr=0.01)
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

model.fit_generator(batches, steps_per_epoch=batches.samples//batch_size // hvd.size(), nb_epoch=10,
                validation_data=valid_batches, validation_steps=valid_batches.samples//batch_size//hvd.size())


if hvd.rank() == 0:
  model_json = model.to_json()
  with open("model.json",'w') as json_file:
    json_file.write(model_json)
  model.save_weights("model_first.h5")
  print("Saved mode in the first step")
