# -*- coding: utf-8 -*-
"""
Created on Wed Dec  8 20:11:56 2021

@author: abhis
"""
# import the necessary packages
#import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

#from keras.optimizers import RMSprop
#from keras.preprocessing.image import ImageDataGenerator
#import cv2
#from keras.models import Sequential
#from keras.layers import Conv2D, Input, ZeroPadding2D, BatchNormalization, Activation, MaxPooling2D, Flatten, Dense,Dropout
#from keras.models import Model, load_model
#from keras.callbacks import TensorBoard, ModelCheckpoint
from sklearn.model_selection import train_test_split
#from sklearn.metrics import f1_score
#from sklearn.utils import shuffle
#import imutils
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
#from tensorflow.keras.applications import MobileNetV2

#from tensorflow.keras.layers import AveragePooling2D
#from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
# from tensorflow.keras.layers import Dense
# from tensorflow.keras.layers import Input
# from tensorflow.keras.models import Model
# from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import os

# initialize the initial learning rate, number of epochs to train for,
# and batch size
INIT_LR = 1e-4
EPOCHS = 1
BS = 32

DIRECTORY = r"C:/Users/abhis/Desktop/TU_Munich_MSCE/advanced-ce-project/dataset-20211208T194250Z-001/dataset"
CATEGORIES = ["with_mask", "without_mask"]

# grab the list of images in our dataset directory, then initialize
# the list of data (i.e., images) and class images
print("[INFO] loading images...")

data = []
labels = []

for category in CATEGORIES:
    path = os.path.join(DIRECTORY, category)
    print(path)
    for img in os.listdir(path):
    	img_path = os.path.join(path, img)
    	image = load_img(img_path, target_size=(224, 224))
    	image = img_to_array(image)
    	image = preprocess_input(image)

    	data.append(image)
    	labels.append(category)

        #print(data)
        
        #print(category)
        
#print(data)
#print(labels)

lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)

print(labels)

data = np.array(data, dtype="float32")
labels = np.array(labels)


(trainX, testX, trainY, testY) = train_test_split(data, labels,
	test_size=0.20, stratify=labels, random_state=42)


#trainY = np.asarray(labels).astype('float32').reshape((-1,1))

#testY = np.asarray(labels).astype('float32').reshape((-1,1))
  
print("d")

aug = ImageDataGenerator(
	rotation_range=20,
	zoom_range=0.15,
	width_shift_range=0.2,
	height_shift_range=0.2,
	shear_range=0.15,
	horizontal_flip=True,
	fill_mode="nearest")



# model =Sequential([
#     Conv2D(100, (3,3), activation='relu', input_shape=(224, 224, 3)),
#     MaxPooling2D(2,2),
    
#     Conv2D(100, (3,3), activation='relu'),
#     MaxPooling2D(2,2),
    
#     Flatten(),
#     Dropout(0.5),
#     Dense(50, activation='relu'),
#     Dense(2, activation='softmax')
# ])


model = keras.Sequential()
model.add(layers.InputLayer(input_shape=(224, 224, 3)))  #3 is for rgb
model.add(layers.Conv2D(100, 3, strides=2, activation="relu"))
model.add(layers.MaxPooling2D(2,2))
model.add(layers.Conv2D(100, 3, strides=2, activation="relu"))
model.add(layers.MaxPooling2D(2,2))
#model.add(Flatten())
model.add(layers.Dense(50, activation='relu'))
model.add(Flatten())
model.add(layers.Dense(2, activation='softmax'))  #2 classes mask or no mask, softmax gives prob for output class
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])  #binary is just fpr 2 classes

model.summary()


history = model.fit(
	aug.flow(trainX, trainY, batch_size=BS),
	steps_per_epoch=len(trainX) // BS,
	validation_data=(testX, testY),
	validation_steps=len(testX) // BS,
	epochs=EPOCHS)



predIdxs = model.predict(testX, batch_size=BS)
predIdxs = np.argmax(predIdxs, axis=1)


print(classification_report(testY.argmax(axis=1), predIdxs,
	target_names=lb.classes_))



# serialize the model to disk
print("[INFO] saving mask detector model...")
model.save("mask_detector.model", save_format="h5")

# plot the training loss and accuracy



history_dict = history.history
history_dict.keys()

acc = history.history['acc'] 
val_acc = history.history['val_acc'] 
loss = history.history['loss'] 
val_loss = history.history['val_loss'] 
print(len(acc))

epochs = range(1, len(acc) + 1)

# Plot the Loss

# "bo" is for "blue dot"
plt.plot(epochs, loss, 'bo', label='Training loss') #TO DO: use the training loss
# b is for "solid blue line"
plt.plot(epochs, val_loss, 'b', label='Validation loss') #TO DO: use validation loss

plt.plot(epochs, acc, 'ro', label='Training acc') #TO DO: use training accuracy
plt.plot(epochs, val_acc, 'r', label='Validation acc')


plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()


#x axis parameter for plotting should come first in plt.plot
plt.show()



