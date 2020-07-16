from sklearn.datasets import fetch_openml
from scipy.ndimage.interpolation import shift
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import numpy as np


# Fetching MNIST Dataset
mnist = fetch_openml('mnist_784', version=1)

# Get the data and target
X, y = mnist["data"], mnist["target"]

# Split the train and test set
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

# Method to shift the image by given dimension
def shift_image(image, dx, dy):
    image = image.reshape((28, 28))
    shifted_image = shift(image, [dy, dx], cval=0, mode="constant")
    return shifted_image.reshape([-1])


# Creating Augmented Dataset
X_train_augmented = [image for image in X_train]
y_train_augmented = [image for image in y_train]

for dx, dy in ((1,0), (-1,0), (0,1), (0,-1)):
     for image, label in zip(X_train, y_train):
             X_train_augmented.append(shift_image(image, dx, dy))
             y_train_augmented.append(label)


# Shuffle the dataset
shuffle_idx = np.random.permutation(len(X_train_augmented))
X_train_augmented = np.array(X_train_augmented)[shuffle_idx]
y_train_augmented = np.array(y_train_augmented)[shuffle_idx]


# Training on augmented dataset
rf_clf_for_augmented = RandomForestClassifier(random_state=42)
rf_clf_for_augmented.fit(X_train_augmented, y_train_augmented)

# Evaluating the model
y_pred_after_augmented = rf_clf_for_augmented.predict(X_test)
score = accuracy_score(y_test, y_pred_after_augmented)
print("Accuracy score after training on augmented dataset", score)
