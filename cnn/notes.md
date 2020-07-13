Maximum number of conv layers is 3 because image gets too small after 3 layers.
Fewer layers gives a faster epoch (9s low at 1 layer compared to 13s low with 3 layers).
Fewer layers also gets 98% accuracy after 2 epochs, and a much higher 1st epoch accuracy, at 94.33%.
Preprocessing with contrast is worse, as it limits the variability of the images. Try in addition to.