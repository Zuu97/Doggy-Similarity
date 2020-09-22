import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

import logging
logging.getLogger('tensorflow').disabled = True

import keras
import numpy as np
from keras.models import model_from_json
from keras.models import Sequential
from keras.layers import Activation, Dense, Input
from util import image_data_generator, load_test_data
from variables import *

class VGG16(object):
    def __init__(self):
        test_classes, test_images = load_test_data()
        train_generator, validation_generator, test_generator = image_data_generator()
        self.test_generator = test_generator
        self.train_generator = train_generator
        self.validation_generator = validation_generator
        self.train_step = self.train_generator.samples // batch_size
        self.validation_step = self.validation_generator.samples // valid_size
        self.test_step = self.test_generator.samples // batch_size
        self.test_classes = test_classes
        self.test_images = test_images

    def model_conversion(self): #VGG16 is not build through sequential API, so we need to convert it to sequential
        vgg_functional = keras.applications.vgg16.VGG16()
        model = Sequential()
        for layer in vgg_functional.layers[:-1]:# remove the softmax in original model. because we have only 3 classes
            layer.trainable = False
            model.add(layer)
        model.add(Dense(dense_1, activation='relu'))
        model.add(Dense(dense_2, activation='relu'))
        model.add(Dense(dense_2, activation='relu'))
        model.add(Dense(num_classes, activation='softmax'))
        model.summary()
        self.model = model

    def train(self):
        self.model.compile(
                          optimizer='Adam',
                          loss='categorical_crossentropy',
                          metrics=['accuracy']
                          )
        self.model.fit_generator(
                          self.train_generator,
                          steps_per_epoch= self.train_step,
                          validation_data= self.validation_generator,
                          validation_steps = self.validation_step,
                          epochs=epochs,
                          verbose=verbose
                        )

    def save_model(self):
        print("Model Saving !")
        model_json = self.model.to_json()
        with open(model_architecture, "w") as json_file:
            json_file.write(model_json)
        self.model.save_weights(model_weights)

    def load_model(self):
        self.model.load_weights(model_weights)
        print("Model Loaded")

        self.model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])


    def predict_VGG16(self):
        Predictions = self.model.predict_generator(self.test_generator,steps=self.test_step)
        P = np.argmax(Predictions,axis=1)
        loss , accuracy = self.model.evaluate_generator(self.test_generator, steps=self.test_step)
        print("test loss : ",loss)
        print("test accuracy : ",accuracy)
        print("Predictions : ",P)

    def run_VGG16(self):
        if os.path.exists(model_weights):
            self.load_model()
        else:
            self.train()
            self.save_model()
        # self.predict_VGG16()

    def feature_extractor(self):
        feature_model = Sequential()
        for layer in self.model.layers[:-1]:# remove the softmax in original model. because we have only 3 classes
            layer.trainable = False
            feature_model.add(layer)
        self.feature_model = feature_model

    def extract_features(self):
        Predictions = self.feature_model.predict(self.test_images)
        print(Predictions.shape)

    def run_feature_model(self):
        self.feature_extractor()
        self.extract_features()

if __name__ == "__main__":
    model = VGG16()
    model.model_conversion()
    model.run_VGG16()
    model.run_feature_model()