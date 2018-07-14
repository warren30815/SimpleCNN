from PIL import Image
from os import listdir
from os import walk
from os.path import isfile, isdir, join
import keras
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Activation, Dropout, Flatten
from keras import backend as K
import numpy as np

#資料參數
img_rows, img_cols, channels = 28, 28, 1
top_model_weights_path = 'top_model_weights.h5'
num_classes = 10
input_shape=(img_rows,img_cols,1)

#預測歷史
class LossHistory(keras.callbacks.Callback):
    def __init__(self,model,x_test, y_test):
        self.model = model
        self.x_test = x_test
        self.y_test = y_test
        self.i=0
    """
    def on_epoch_end(self, epoch, logs=None):
        print("\ntime:"+str(i))
    """
    def on_train_end(self, logs=None):
        y_test_pred = np.argmax(self.model.predict(self.x_test), axis=1)
        error=0
        for i in range(len(y_test)):
            if(y_test_pred[i]!=y_test[i]):
                error+=1
        print("\n104503530's Test Accurance: ",(len(y_test)*1.0-(error*1.0))/len(y_test))

#讀取訓練資料路徑
train_path = "./image"
train_datagen = ImageDataGenerator(     #training data generator
                rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
                width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
                height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
                zoom_range=0.1,
                fill_mode='nearest',
                horizontal_flip=False,  # randomly flip images
                vertical_flip=False,
                )  
files = listdir(train_path)
x_data=np.array([])
data_number=0
#讀取資料夾內所有檔案
for root, dirs, files in walk(train_path):
    for f in files:
        if data_number==0:
            fullpath = join(root, f)
            im = Image.open(fullpath)
            x_data = (np.array(im) / 255).reshape(1,28,28)  # 讀取資料時順便做資料正規化
            #print(fullpath)
            #print(x_data.shape)
            data_number += 1
        else:
            fullpath = join(root, f)
            im = Image.open(fullpath)
            im = (np.array(im)/255).reshape(1,28,28)
            x_data = np.vstack((x_data,im)) # 讀取資料時順便做資料正規化
            #print(fullpath)
            #print(x_data.shape)
            data_number += 1
x_data=x_data.reshape(data_number,img_rows,img_cols,1) #調整資料格式
#建立label
y_data=[]
for k in range(0,49,1):
    for i in range(0,10,1):
        for j in range(0,5,1):
            y_data.append(i)

#one hot encoding
y_train = keras.utils.to_categorical(y_data, num_classes)

train_flow = train_datagen.flow(     #augmentation of train data
                x_data,
                y_train,
                batch_size=20,    
                shuffle=True) 
#讀取測試資料
test_path = "./test_image"
files = listdir(test_path)
x_test=[]
test_number=0
for root, dirs, files in walk(test_path):
    for f in files:
        fullpath = join(root, f)
        im = Image.open(fullpath)
        p = np.array(im)/255
        x_test.append(p)
        test_number+=1
x_test=np.array(x_test)
x_test=x_test.reshape(test_number,img_rows,img_cols,1)

#建立test_label
y_test=[]
for k in range(0,34,1):
    for i in range(0,10,1):
        for j in range(0,5,1):
            y_test.append(i)

#建立模型
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
    # x = Conv2D(128, (2, 2), strides=(2, 2), padding='valid', name='conv4')(x)
    # x = Activation('relu', name='relu_conv4')(x)
    # x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool3')(x)

    model = Model(img_input, x, name='net')
    return model

model = Net()
#Freeze the layers which you don't want to train
# for layer in model.layers:
#     layer.trainable = False

top_model = model.output
top_model = Dropout(0.5, name='drop9')(top_model)
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

top_model.load_weights(top_model_weights_path)
#設定model參數
top_model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer='adam',
              metrics=['accuracy'])
#設定訓練紀錄
history = LossHistory(model, x_test, y_test)
#開始訓練
top_model.fit_generator(
            generator=train_flow,
            steps_per_epoch=30000 // 20,   
            verbose=2, 
            validation_data=train_flow,
            validation_steps=5000 // 32,
            epochs=30,
            callbacks=[history]
            )

#pre=model.predict(test_data)
#print(np.argmax(pre, axis=1))