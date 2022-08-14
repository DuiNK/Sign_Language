import os
import warnings
import cv2
import keras
import matplotlib.pyplot as plt
import matplotlib.style as style
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from PIL import Image
from keras import models, layers, optimizers
from keras.applications.vgg16 import VGG16
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Dense, Dropout, Flatten
from keras.models import Model
from keras.preprocessing import image as image_utils
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
#vggmodel = VGG16(weights='imagenet', include_top=True)
rgb = False
gestures_map = {'0': 0,
                '1': 1,
                '2': 2,
                '3': 3,
                '4': 4,
                '5': 5,
                '6': 6,
                '7': 7,
                '8': 8,
                '9': 9,
                '10': 10,
                '11': 11,
                '12': 12,
                '13': 13,
                '14': 14,
                '15': 15,
                '16': 16,
                '17': 17,
                '18': 18,
                '19': 19,
                '20': 20,
                '21': 21,
                '22': 22,
                '23': 23,
                '24': 24,
                '25': 25}

gesture_names = {0: 'A',
                 1: 'B',
                 2: 'C',
                 3: 'D',
                 4: 'E',
                 5: 'F',
                 6: 'G',
                 7: 'H',
                 8: 'I',
                 9: 'J',
                 10: 'K',
                 11: 'L',
                 12: 'M',
                 13: 'N',
                 14: 'O',
                 15: 'P',
                 16: 'Q',
                 17: 'R',
                 18: 'S',
                 19: 'T',
                 20: 'U',
                 21: 'V',
                 22: 'W',
                 23: 'X',
                 24: 'Y',
                 25: 'Z'}

def process_data(X_data, y_data):
    X_data = np.array(X_data, dtype = 'float32')
    if rgb:
        pass
    else:
        X_data = np.stack((X_data,)*3, axis=-1)
    X_data /= 255
    y_data = np.array(y_data)
    y_data = to_categorical(y_data)
    return X_data, y_data
#data_dir_train = '/content/Sign_Language/data/train'
#data_dir_val = '/content/Sign_Language/data/test'
# Create a dataset
batch_size = 32
img_height = 50
img_width = 50

rootpath = '/content/Sign_Language/data/test'
list = os.listdir(rootpath) #列出文件夹下所有的目录与文件
#设定图像宽高
imgwidth = 50
imgheight = 50

imgdata_test = []
imgtag_test = []
print(len(list))
for i in range(len(list)):
    #对于子目录进行处理
    # print(i)
    currentpath = rootpath+ "/" + list[i]
    currentlist = os.listdir(currentpath)
    print(list[i])
    for j in range(len(currentlist)):
        #图像位置
        imgpath = currentpath + "/" + currentlist[j]
        #有后缀为db的文件
        if currentlist[j][-3:] == "jpg":
            img = cv2.imread(imgpath,0)
            img = cv2.resize(img,(imgwidth,imgheight))
            #name = gestures_map[os.path.basename(imgpath)]
            Path = os.path.dirname(imgpath)
            name_test = os.path.basename(Path)
            #print(img)
            #print(name_test)
            imgtag_test.append(gestures_map[name_test])
            imgdata_test.append(img)
#    imgdata_test, imgtag_test = process_data(imgdata_test, imgtag_test)
#imgtag_test = np.array(imgtag_test)
#imgdata_test = np.array(imgdata_test)
'''
imgtag = imgtag.reshape(imgtag.shape[0],1)
#增加一维灰度维
imgdata = imgdata.reshape(imgdata.shape[0],imgdata.shape[1],imgdata.shape[2],1)
print(imgdata.shape,imgtag.shape)
#存储
'''
rootpath_train = '/content/Sign_Language/data/train'
list_train = os.listdir(rootpath_train) #列出文件夹下所有的目录与文件
#设定图像宽高
imgwidth = 50
imgheight = 50

imgdata_train = []
imgtag_train = []
print(len(list_train))
for i in range(len(list_train)):
    #对于子目录进行处理
    # print(i)
    currentpath = rootpath_train+ "/" + list[i]
    currentlist = os.listdir(currentpath)
    print(list[i])
    for j in range(len(currentlist)):
        #图像位置
        imgpath = currentpath + "/" + currentlist[j]
        #有后缀为db的文件
        if currentlist[j][-3:] == "jpg":
            img = cv2.imread(imgpath,0)
            img = cv2.resize(img,(imgwidth,imgheight))
            Path = os.path.dirname(imgpath)
            name_train = os.path.basename(Path)
            imgtag_train.append(name_train)
            imgdata_train.append(img)
    
#imgtag_train = np.array(imgtag_train)
#imgdata_train = np.array(imgdata_train)

# Dinh nghia cac bien

image_path = '/content/drive/MyDrive/datatest'
models_path = '/content/gdrive/MyDrive/Colab Notebooks/saved_model2.hdf5'

imageSize = 50
'''
image = tf.keras.preprocessing.image.load_img(image_path)

# Ham xu ly anh resize ve 224x224 va chuyen ve numpy array
def process_image(path):
    img = Image.open(path)
    img = img.resize((imageSize, imageSize))
    img = np.array(img)
    return img
'''
imgdata_train, imgtag_train = process_data(imgdata_train, imgtag_train)
imgdata_test, imgtag_test = process_data(imgdata_test, imgtag_test)
# Load du lieu vao X va Y
X_train = imgdata_train
X_test = imgdata_test
y_train = imgtag_train
y_test = imgtag_test
# Phan chia du lieu train va test theo ty le 80/20
#X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size = 0.2, random_state=12, stratify=y_data)

# Dat cac checkpoint de luu lai model tot nhat
model_checkpoint = ModelCheckpoint(filepath=models_path, save_best_only=True)
early_stopping = EarlyStopping(monitor='val_acc',
                               min_delta=0,
                               patience=10,
                               verbose=1,
                               mode='auto',
                               restore_best_weights=True)

# Khoi tao model
model1 = VGG16(weights='imagenet', include_top=False, input_shape=(imageSize, imageSize, 3))
optimizer1 = tf.keras.optimizers.Adam()
base_model = model1

# Them cac lop ben tren
x = base_model.output
x = Flatten()(x)
x = Dense(64, activation='relu', name='fc1')(x)
x = Dense(64, activation='relu', name='fc2')(x)
#x = Dense(128, activation='relu', name='fc2a')(x)
x = Dense(64, activation='relu', name='fc3')(x)
x = Dropout(0.2)(x)
x = Dense(32, activation='relu', name='fc4')(x)

predictions = Dense(26, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)

# Dong bang cac lop duoi, chi train lop ben tren minh them vao
for layer in base_model.layers:
    layer.trainable = False

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=150, batch_size=32, validation_data=(X_test, y_test), verbose=1,
          callbacks=[early_stopping, model_checkpoint])

# Luu model da train ra file
model.save('/content/gdrive/MyDrive/Colab Notebooks/mymodel2.h5')
new_model = keras.models.load_model('/content/gdrive/MyDrive/Colab Notebooks/mymodel2.h5')



