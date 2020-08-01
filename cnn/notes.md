Maximum number of conv layers is 3 because image gets too small after 3 layers.
Fewer layers gives a faster epoch (9s low at 1 layer compared to 13s low with 3 layers).
Fewer layers also gets 98% accuracy after 2 epochs, and a much higher 1st epoch accuracy, at 94.33%.
Preprocessing with contrast is worse, as it limits the variability of the images. Try in addition to.

Used tensorflow for CNN.
Due to our approach being fairly broad, I didn't go much into tweaking many parameters, and tensorflow handles a lot of the optimisations out of the box.
Added onto this, I wanted to cover CNN fairly quickly as it's often the main technique for classifying images, and we wanted to look into other things.
There also seemed to be quite a lot of adversarial examples, but enough that they might not actually throw the model off.
Convolves over each input, to learn features based on the adjustment of filter matrices used.
Simple dataset, few pre-processing techniques seemed help.
If the test dataset were less sanitised or we were testing our own set, introducing noise and other pre-processing techiques would have helped.

## 3 Epoch Results (no pre-processing)

1 Conv Layer: 0.9803 acc, 0.0602 loss, 21 secs
              0.9817 acc, 0.0553 loss, 21 secs
              0.9832 acc, 0.0464 loss, 21 secs
         avg: 0.9817 acc, 0.0540 loss, 21 secs
2 Conv Layers: 0.9854 acc, 0.0438 loss, 30 secs
               0.9884 acc, 0.0352 loss, 30 secs
               0.9907 acc, 0.0307 loss, 30 secs
          avg: 0.9881 acc, 0.0366 loss, 30 secs
3 Conv Layers: 0.9819 acc, 0.0615 loss, 32 secs
               0.9799 acc, 0.0653 loss, 30 secs
               0.9809 acc, 0.0645 loss, 31 secs
          avg: 0.9809 acc, 0.0638 loss, 31 secs

For all of these, loss tends to change from ~0.3 swiftly down to <0.1 within 1 or two epochs, and accuracy goes from around 0.93 to 0.975 within a similar time.
Validation loss and accuracy improve much less, with val_loss already below 0.1 in the first epoch, and val_accuracy above 0.97 in the first epoch often.

## 3 Epoch Results (sharpening)

1 Conv Layer: 0.9788 acc, 0.1516 loss, 22 secs
              0.9676 acc, 0.1632 loss, 21 secs
              0.9759 acc, 0.1508 loss, 21 secs
         avg: 0.9741 acc, 0.1552 loss, 21 secs
2 Conv Layers: 0.9854 acc, 0.1118 loss, 30 secs
               0.9858 acc, 0.0946 loss, 30 secs
               0.9882 acc, 0.0868 loss, 30 secs
          avg: 0.9881 acc, 0.0977 loss, 30 secs
3 Conv Layers: 0.9763 acc, 0.1643 loss, 30 secs
               0.9795 acc, 0.1541 loss, 30 secs
               0.9769 acc, 0.1436 loss, 30 secs
          avg: 0.9776 acc, 0.1540 loss, 30 secs

Much less change in loss, acc, val_loss, val_acc across epochs this time.
**Loss is up between 2 and 3 times as high** - may be due to training the model for a feature it'll never see in the test set?

## 10 Epoch Results (no pre-processing)

### 1 Layer

Maximum val_acc 0.9873, didn't change much after epoch 6.
Hits 0.98 within 2 epochs.

### 2 Layers

Hits 0.99 on some epochs. 2 layer config still on top.

### 3 Layers

Similar performance to 1 layer.

Not much benefit in having more than a few epochs.


No improvement with sharpening added across the board, just higher loss as usual with a larger training set.
