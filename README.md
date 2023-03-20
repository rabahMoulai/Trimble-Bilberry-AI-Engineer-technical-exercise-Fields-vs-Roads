# Trimble-Bilberry : AI_Engineer_technical_exercise (Image Binary Classification : Field vs Road )


## Objective

This is a binary image classification project using Convolutional Neural Networks ( inspired by AlexNet) and TensorFlow Keras API on Python 3.
The objective of this task is to train a neural network to classify images of roads and fields.


## Approach



## Insights from Data

the first thing we notice when exploring our dataset is that the dataset is unbalanced and image resolution is not stable.
we notice that the road class is in majority. To solve this problem, I chose to increase the data of the minority class. (over sampling)

A small rotation is performed to equalize the number of examples in the majority class.
Before data augmentation some transformations are necessary I resize the images to (124,124) and normalize the dataset by dividing by 255 to convert the pixels in range [0,255] to range [0,1]

## Transformations (Preprocessing)

As the test set is provided without labels, I chose to divide the training set into a test set to evaluate the model and a set for training and validation.
(80%training 20%test) Note that the 10 images provided with the test set will be used as inferences for the prediction

Even after balancing our data set, the data set remains very small
To improve this, I will use ImageDataGenerator from tf.keras to augment the images with different augmentation techniques like rotation, small Zoom and a horizontal flip.


Other transformations were considered, such as rotations, vertical flipping, and applying a grayscale filter. These were not chosen, however, as they produce changes to the images that compromise their defining features. I also chose not to apply grayscale filtering, because the color gives important clues to the classification (e.g. fields are likely to have more green, roads likely to have more gray).


However, the main benefit of using the Keras ImageDataGenerator class is that it is designed to provide real-time data augmentation. Meaning it is generating augmented images on the fly while the model is still in the training stage.









## CNN Architecture













## Results

we evaluate our model with the test set but since it is a binary classifier we will not only evaluate the accuracy but also the ROC curve and the confusion matrix.

the accuracy on test set is between (85% to 94.4 %). this variation is due to the random factor when increasing the data by the image generator.

The confusion matrix show that only 5 of the 39 are badly classified and i have AUC = 0.97.

inferences images and their predictions can be viewed in the "predictions.png" image The neural network typically predicts correctly 8 to 10 of the 10 inferences images.

