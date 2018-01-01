#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  3 07:22:13 2017

@author: tanmoy
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import keras 
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPool2D
from keras.layers import Flatten
from keras.layers import Dense  


#Initialing CNN
Classifier=Sequential()

#Convolution Step:
Classifier.add(Convolution2D(filters=32,kernel_size=(3,3),input_shape = (64, 64, 3),activation='relu'))

#Pooling Step:
Classifier.add(MaxPool2D(pool_size=(2,2)))

## Adding More Convulational Layer
Classifier.add(Convolution2D(filters=32,kernel_size=(3,3),activation='relu'))
Classifier.add(MaxPool2D(pool_size=(2,2)))

#Flattening:
Classifier.add(Flatten())

#Connecting with ANN:
Classifier.add(Dense(units=128, activation='relu'))
Classifier.add(Dense(units=1, activation='sigmoid'))
Classifier.compile(optimizer='adam',loss='binary_crossentropy', metrics=['accuracy'])

#Data Preprocessing and Loading from Hard drive
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory('dataset/training_set',target_size=(64,64),batch_size=32,class_mode='binary')

test_set = test_datagen.flow_from_directory(
        'dataset/test_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

Classifier.fit_generator(
        training_set,
        steps_per_epoch=8000,
        epochs=2,
        validation_data=test_set,
        validation_steps=2000)

#Single Image Prediction
single_image=test_datagen.flow_from_directory('dataset/single_prediction',target_size=(64,64),class_mode='binary')


pred=Classifier.predict(single_image)

