import cnn
import tensorflow as tf


def main():
    (train_images, train_labels), (test_images, test_labels) = cnn.init()
    train_images, test_images = cnn.preprocess_normalise(train_images, test_images)
    train_images += cnn.preprocess_contrast(train_images)
    model = cnn.create_model(1)
    model.summary()
    model.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])
    history = model.fit(train_images, train_labels, epochs=3, 
                        validation_data=(test_images, test_labels))
    cnn.plot_history(history)
    test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
    print(test_acc)


if __name__ == "__main__":
    main()