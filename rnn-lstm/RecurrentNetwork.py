import tensorflow as tf
import tensorflow.keras.layers as KL
from sklearn.metrics import confusion_matrix
from mlxtend.plotting import plot_confusion_matrix
import numpy as np
## Dataset
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train/255.0, x_test/255.0

## Model
inputs = KL.Input(shape=(28, 28))
# For RNN
#x = KL.RNN(64, activation ='relu')(inputs) 

# For LSTM
x = KL.LSTM(128)(inputs)#Adds an LSTM with 128 Internal units

outputs = KL.Dense(10, activation="softmax")(x)

model = tf.keras.models.Model(inputs, outputs)
model.summary()
model.compile(optimizer="adamax", #try with adamax and rmsprop too see slight variations in results
                loss="sparse_categorical_crossentropy",
                metrics=["acc"])
model.fit(x_train, y_train, epochs=5)
test_loss, test_acc = model.evaluate(x_test, y_test)
print("Loss: {0} - Acc: {1}".format(test_loss, test_acc))
predictions = model.predict(x_test)
confusion = confusion_matrix(y_test, np.argmax(predictions,axis=1),labels=[0,1,2,3,4,5,6,7,8,9])
print(confusion)
plot_confusion_matrix(confusion)