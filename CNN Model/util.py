import tensorflow as tf
import numpy as np
import keras
import os
import numpy as np
import cv2 as cv
from sklearn.utils import shuffle
from keras.preprocessing.image import ImageDataGenerator

from variables import*

def image_data_generator():
    train_datagen = ImageDataGenerator(
                                    rescale = rescale,
                                    rotation_range = rotation_range,
                                    shear_range = shear_range,
                                    zoom_range = zoom_range,
                                    width_shift_range=shift_range,
                                    height_shift_range=shift_range,
                                    horizontal_flip = True,
                                    validation_split= val_split
                                    )
    test_datagen = ImageDataGenerator(rescale = rescale)


    train_generator = train_datagen.flow_from_directory(
                                    train_dir,
                                    target_size = target_size,
                                    color_mode = color_mode,
                                    batch_size = batch_size,
                                    classes = classes,
                                    shuffle = True)

    validation_generator = test_datagen.flow_from_directory(
                                    train_dir,
                                    target_size = target_size,
                                    color_mode = color_mode,
                                    batch_size = valid_size,
                                    classes = classes,
                                    shuffle = True)

    test_generator = test_datagen.flow_from_directory(
                                    test_dir,
                                    target_size = target_size,
                                    color_mode = color_mode,
                                    batch_size = batch_size,
                                    classes = classes,
                                    shuffle = False)

    return train_generator, validation_generator, test_generator

def load_test_data():
    if not os.path.exists(test_data_path):
        print("Test Images Saving")
        images = []
        classes = []
        dog_folders = os.listdir(test_dir)
        for label in list(dog_folders):
            label_dir = os.path.join(test_dir, label)
            label_images = []
            for img_name in os.listdir(label_dir):
                img_path = os.path.join(label_dir, img_name)
                img = cv.imread(img_path)
                img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
                img = cv.resize(img, target_size)
                images.append(img)
                classes.append(int(label))
        images = np.array(images)
        classes = np.array(classes)
        np.savez(test_data_path, name1=images, name2=classes)
    else:
        print("Test Images Loaded")
        data = np.load(test_data_path, allow_pickle=True)
        images = data['name1']
        classes = data['name2']

    classes, images = shuffle(classes, images)
    return classes, images