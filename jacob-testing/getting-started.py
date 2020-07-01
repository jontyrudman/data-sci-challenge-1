import tensorflow as tf
import matplotlib.pyplot as plt

from wrapper import ModelWrapper

def manual():
    # build the model
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten()) # turn the images from a 28x28 grid to a 1x784
    model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax)) # the output layer

    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    # train the model
    model.fit(x=x_train, y=y_train, epochs=5)

    _, test_acc = model.evaluate(x=x_test, y=y_test)
    print(f"\nTest accuracy = {test_acc}")

    return model.predict(x_test)

def wrapped():
    layers = [
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation=tf.nn.relu),
        tf.keras.layers.Dense(128, activation=tf.nn.relu),
        tf.keras.layers.Dense(10, activation=tf.nn.softmax)
    ]

    model = ModelWrapper(layers)
    model.build()

    model.train(x_train, y_train)

    accuracy = model.getAccuracy(x_test, y_test)
    print(f"\nTest accuracy = {accuracy}")

    return model.getPredictions(x_test)

# load the dataset
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# show the first image in the dataset
# plt.imshow(x_train[0], cmap="gray")
# plt.show()

# normalise the datasets
x_train, x_test = x_train / 255.0, x_test / 255.0

predictions = wrapped()
print(tf.argmax(predictions[1000])) # should be 9
# show the corresponding image
plt.imshow(x_test[1000], cmap="gray")
plt.show()
