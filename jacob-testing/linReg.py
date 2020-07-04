def linearRegression(xValues, yValues):
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

        print(i)

    def predictor(x):
        print('Created a generating function')
        return (w2 * (x ** 2)) + (w1 * x) + w0

    return predictor

func = linearRegression(
    [-100, -10, -3, 0, 1, 2, 3, 4, 10],
    [9901, 91, 7, 1, 3, 7, 13, 21, 111, 10101])

print('Test 4: ', func(4))