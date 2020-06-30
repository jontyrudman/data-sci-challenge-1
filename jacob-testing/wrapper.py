import tensorflow as tf

class ModelWrapper:
    """A wrapper for creating tensorflow sequential models."""

    model = None

    def __init__(self, layers):
        """
        Constructor for the object.

        Args:
        - layers: A list of tensorflow layer objects for the model.
        """

        self.layers = layers

    def build(self, compile_opt = "adam", compile_loss="sparse_categorical_crossentropy", compile_metrics=["accuracy"]):
        """
        Adds all specified layers to the model and then compiles it.

        Args:
        - (Optional) compile_opt: The optimiser to be used.
        - (Optional) compile_loss: The loss function to be used.
        - (Optional) compile_metrics: The metrics used to test the model.
        """

        self.model = tf.keras.models.Sequential()
        for layer in self.layers:
            self.model.add(layer)

        self.model.compile(optimizer=compile_opt, loss=compile_loss, metrics=compile_metrics)

    def train(self, x, y, epochs):
        """
        A simple way to train the model.

        Args:
        - x: The input training data.
        - y: The output training data.
        - epochs: The number of iterations of training.
        """

        self.model.fit(x=x, y=y, epochs=epochs)

    def getAccuracy(self, x, y) -> float:
        """
        A simple way to find how accurate the model is.

        Args:
        - x: The input test data.
        - y: The input test data.

        Returns:
        The accuracy as a float.
        """

        _, accuracy = self.model.evaluate(x=x, y=y)

        return accuracy

    def getPredictions(self, x):
        """
        Uses the model to make predictions on the input data.

        Args:
        - x: The input test data.

        Returns
        A numpy-array of predictions
        """

        return self.model.predict(x)