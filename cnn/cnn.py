import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import cv2 


OUTPUTS = ['0', '1', '2', '3', '4', '5',
               '6', '7', '8', '9']


def init():
    # For any graphs needed
    matplotlib.use('tkagg')
    # Load dataset
    (train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()
    return (train_images, train_labels), (test_images, test_labels)


def preprocess_normalise(train, test):
    train, test = train / 255.0, test / 255.0
    return train, test

def reshape(train, test):
    train = train.reshape(train.shape[0], 28, 28, 1)
    test = test.reshape(test.shape[0], 28, 28, 1)
    return train, test


def preprocess_contrast(input_imgs):
    imgs = []
    for i in range(len(input_imgs)):
        np_img = np.array(input_imgs[i])
        np_img[np_img > 0.5] = 1
        np_img[np_img <= 0.5] = 0
        imgs += [np_img]
    return imgs


def create_model(size):
    model = models.Sequential()
    model.add(layers.Conv2D(28, (3, 3), activation='relu', input_shape=(28,28, 1)))
    model.add(layers.MaxPooling2D((2, 2)))
    num_channels = 28

    # Size validation
    if size < 1: size = 1
    elif size > 5: size = 5

    for i in range(size - 1):
        num_channels += 10
        model.add(layers.Conv2D(num_channels, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Flatten())
    model.add(layers.Dense(40, activation='relu'))
    # 10 output classes
    model.add(layers.Dense(10))

    return model


def plot_history(history):
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0.5, 1])
    plt.legend(loc='lower right')
    plt.show()