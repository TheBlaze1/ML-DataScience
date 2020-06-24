# -*- coding: utf-8 -*-
"""
Created on Sun May 24 16:59:28 2020

@author: unibl
"""


from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
#Initializing the CNN (Convolutional Neural Network)
classifier = Sequential()
classifier.add(Convolution2D(32,3,3, input_shape = (64,64,3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2,2)))
classifier.add(Convolution2D(32,3,3, activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2,2)))
# Flattening
classifier.add(Flatten())
# Full Connection
classifier.add(Dense(output_dim= 128, activation = 'relu'))
classifier.add(Dense(output_dim= 1, activation = 'sigmoid'))
#Compiling CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
# Part 2  Fitting Images to CNN
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)
training_set = train_datagen.flow_from_directory(
        'dataset/training_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')
test_set = test_datagen.flow_from_directory(
        'dataset/test_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')
classifier.fit(
        training_set,
        steps_per_epoch=8000,
        epochs=25,
        validation_data= test_set,
        validation_steps=2000)
# Making new prediction 
import numpy as np
from keras.preprocessing import image

test_image = image.load_img('dataset/single_prediction/cat_or_dog_1.jpg', target_size=(64, 64))
test_image = image.img_to_array(test_image)
test_iamge = np.expand_dims(test_image , axis = 0)
result = classifier.predict(test_iamge)
training_set.class_indices
if result[0][0] == 1:
    prediction = 'dog'
else:
    prediction = 'cat'
from keras.models import load_model 
# Saving the model
# Making new prediction 
import numpy as np
from keras.preprocessing import image
from keras.models import load_model 
test_image = image.load_img('dataset/single_prediction/cat_or_dog_1.jpg', target_size=(64, 64))
test_image = image.img_to_array(test_image)
test_iamge = np.expand_dims(test_image , axis = 0)
model = load_model('my_model_CNN.h5')# Loading Model
result = model.predict(test_iamge)

if result[0][0] == 1:
    prediction = 'dog'
else:
    prediction = 'cat'
classifier.save('my_model_CNN.h5') # saving model


