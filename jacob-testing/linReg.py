import argparse
import tensorflow as tf

from wrapper import ModelWrapper


def linearRegression(xValues, yValues, verbose=False):
    if len(xValues) != len(yValues) or len(xValues) == 0 or len(yValues) == 0:
        return None

    epochs = 100000

    w0 = 0.0
    w1 = 0.0
    w2 = 0.0

    alpha = 0.00000001

    for i in range(epochs):
        for j in range(len(xValues)):
            prediction = w0 + (w1 * xValues[j]) + (w2 * (xValues[j] ** 2))

            difference = yValues[j] - prediction

            w0 += alpha * difference
            w1 += alpha * difference * xValues[j]
            w2 += alpha * difference * (xValues[j] ** 2)

        if verbose:
            print(i)

    def predictor(x):
        return (w2 * (x ** 2)) + (w1 * x) + w0

    if verbose:
        print('Created a generating function')

    return predictor


parser = argparse.ArgumentParser()
parser.add_argument(
    '--accuracy', help='Optimise for accuracy', action='store_true')
parser.add_argument('--time', help='Optimise for time', action='store_true')

args = parser.parse_args()

mnist = tf.keras.datasets.mnist
(x_train, y_train), (_, _) = mnist.load_data()
x_train = x_train / 255.0

options = [x for x in range(151) if x % 10 == 0]

print(options)

yValues = []

for option in options:
    layers = [
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(option, activation=tf.nn.relu),
        tf.keras.layers.Dense(10, activation=tf.nn.softmax)
    ]

    model = ModelWrapper(layers)
    model.build()

    output = []

    if args.accuracy:
        (_, output) = model.benchmark(x_train, y_train)
    elif args.time:
        (output, _) = model.benchmark(x_train, y_train)

    average = sum(output) / len(output)

    yValues.append(average)

maximiser = linearRegression(options, yValues)

allOut = list(map(maximiser, range(151)))

print(f'The optimum number of logits/neurons for the hidden layer is {tf.argmax(allOut)}')
