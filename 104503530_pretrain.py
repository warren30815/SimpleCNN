from PIL import Image
from os import listdir
from os import walk
from os.path import isfile, isdir, join
from keras.datasets import mnist
import keras
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Activation, Dropout, Flatten
from keras import backend as K
from keras.utils import np_utils
import numpy as np

img_rows, img_cols, channels = 28, 28, 1
classes = 10
top_model_weights_path = 'top_model_weights.h5'

#create model

def Net(pooling=None):

    img_input = Input(shape=(img_rows, img_cols, channels))
    x = Conv2D(32, (2, 2), strides=(2, 2), padding='valid', name='conv1')(img_input)
    x = BatchNormalization()(x)    
    x = Activation('relu', name='relu_conv1')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool1')(x)
    x = Conv2D(64, (2, 2), strides=(2, 2), padding='valid', name='conv2')(x)
    x = BatchNormalization()(x)    
    x = Activation('relu', name='relu_conv2')(x)
    # x = Conv2D(128, (2, 2), strides=(2, 2), padding='valid', name='conv3')(x)
    # x = Activation('relu', name='relu_conv3')(x)
    # x = Conv2D(64, (2, 2), strides=(2, 2), padding='valid', name='conv4')(x)
    # x = Activation('relu', name='relu_conv4')(x)
    # x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool3')(x)
    model = Model(img_input, x, name='net')
    return model

model = Net()
top_model = model.output
# top_model = Dropout(0.2, name='drop9')(top_model)
top_model = Flatten()(top_model)
top_model = Dense(600)(top_model)
top_model = BatchNormalization()(top_model)
top_model = Activation('relu', name='relu_conv9')(top_model)
# top_model = Dense(600)(top_model)
# top_model = Activation('relu', name='relu_conv10')(top_model)
top_model = Dense(10)(top_model)
top_model = BatchNormalization()(top_model)
top_model = Activation('softmax', name='loss')(top_model)
top_model = Model(model.input, top_model, name='top_net')
#設定model參數
top_model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer='adam',
              metrics=['accuracy'])

#process data
def load_data():
    # load data
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    # normalize inputs from 0-255 to 0-1
    X_train = X_train / 255
    X_test = X_test / 255
    # one hot encode outputs
    y_train = np_utils.to_categorical(y_train, classes)
    y_test = np_utils.to_categorical(y_test, classes)
    return (X_train, y_train), (X_test, y_test)

(X_train, y_train), (X_test, y_test) = load_data()
X_train = np.expand_dims(X_train, axis=3)
X_test = np.expand_dims(X_test, axis=3)
#開始訓練
top_model.fit(X_train, y_train,
          epochs=15,
          batch_size=100,
          verbose=2,
          validation_data=(X_test, y_test),
          shuffle=True)
top_model.save_weights(top_model_weights_path)
#pre=model.predict(test_data)
#print(np.argmax(pre, axis=1))