# Code for gaussian filter and image enhancing.

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import cv2 

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

img = x_train[2]
plt.imshow(img, cmap="gray")
plt.show()

blur = cv2.GaussianBlur(img,(5,5),0)
plt.imshow(blur, cmap="gray")
plt.show()

np_img = np.array(blur)
np_img[np_img > 0.46] = 1

enhanced = blur < np_img
plt.imshow(enhanced, cmap="gray")
plt.show()
