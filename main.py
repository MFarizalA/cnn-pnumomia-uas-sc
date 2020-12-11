from tensorflow.keras.layers import Flatten, Conv2D, MaxPool2D, Dense, Input
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import OneHotEncoder

import os
import numpy as np
import cv2

def readNormal(normPath):
    norm_files = np.array(os.listdir(normPath))
    norm_labels = np.array(['normal'] * len(norm_files))

    norm_images = []
    for image in norm_files:
        image = cv2.imread(normPath + image)
        image = cv2.resize(image, dsize=(200, 200))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        norm_images.append(image)

    norm_images = np.array(norm_images)

    return norm_images, norm_labels


def readPneumonia(pnumPath):
    pneu_files = np.array(os.listdir(pnumPath))
    pneu_labels = np.array([pneu_file.split('_')[1] for pneu_file in pneu_files])

    pneu_images = []
    for image in pneu_files:
        image = cv2.imread(pnumPath + image)
        image = cv2.resize(image, dsize=(200, 200))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        pneu_images.append(image)

    pneu_images = np.array(pneu_images)

    return pneu_images, pneu_labels

def training(X_train):
    return Sequential([
        Conv2D(16, (3, 3), activation='relu', strides=(1, 1), padding='same', input_shape=[X_train.shape[1], X_train.shape[2], 1]),
        Conv2D(32, (3, 3), activation='relu', strides=(1, 1), padding='same'),
        MaxPool2D((2, 2)),
        Conv2D(16, (2, 2), activation='relu', strides=(1, 1), padding='same'),
        Conv2D(32, (2, 2), activation='relu', strides=(1, 1), padding='same'),
        MaxPool2D((2, 2)),
        Conv2D(16, (1, 1), activation='relu', strides=(1, 1), padding='same'),
        Conv2D(32, (1, 1), activation='relu', strides=(1, 1), padding='same'),
        MaxPool2D((2, 2)),
        Flatten(),
        Dense(100, activation='relu'),
        Dense(50, activation='relu'),
        Dense(25, activation='relu'),
        Dense(3, activation='softmax'),
    ])

def imageDataGenerator (X_train):
    generate =  ImageDataGenerator(
        rotation_range = 15,
        zoom_range = 0.5,
        width_shift_range = 0.5,
        height_shift_range = 0.5,
        )
    generate.fit(X_train)

    return generate.flow(X_train, y_train_one_hot, batch_size=32)

if __name__ == '__main__':

    normPathTrain = 'chest_xray/train/NORMAL/'
    pnumPathTrain = 'chest_xray/train/PNEUMONIA/'
    normPathTest ='chest_xray/test/NORMAL/'
    pnumPathTest ='chest_xray/test/PNEUMONIA/'

    train_norm_images, train_norm_labels = readNormal(normPathTrain)
    train_pnum_images, train_pnum_labels = readPneumonia(pnumPathTrain)

    X_train = np.append(train_norm_images, train_pnum_images, axis=0)
    y_train = np.append(train_norm_labels, train_pnum_labels)

    test_norm_images, test_norm_labels = readNormal(normPathTest)
    test_pnum_images, test_pnum_labels = readPneumonia(pnumPathTest)

    X_test = np.append(test_norm_images, test_pnum_images, axis=0)
    y_test = np.append(test_norm_labels, test_pnum_labels)

    y_train = y_train[:, np.newaxis]
    y_test = y_test[:, np.newaxis]

    one_hot_encoder = OneHotEncoder(sparse=False)

    y_train_one_hot = one_hot_encoder.fit_transform(y_train)
    y_test_one_hot = one_hot_encoder.transform(y_test)

    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)

    model = training(X_train)

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])

    model.fit_generator(imageDataGenerator(X_train), epochs=15, validation_data=(X_test, y_test_one_hot))

    model.save('pneumonia-uas-sc-cnn.h5')