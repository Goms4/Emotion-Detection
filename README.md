# Emotion Detection Model
This repository contains an emotion detection model that can identify seven different emotions from facial expressions in images. The model is trained using TensorFlow and Keras, and a simple GUI is built using Tkinter to allow users to upload images and detect emotions.

## Model Architecture
The emotion detection model is a convolutional neural network (CNN) designed to classify facial expressions into one of seven categories: Angry, Disgust, Fear, Happy, Neutral, Sad, and Surprise. The model architecture includes several convolutional layers followed by batch normalization, activation, max-pooling, and dropout layers to prevent overfitting. The final dense layers produce a softmax output for classification.

## Data Augmentation
To enhance the training process, data augmentation techniques are used. These techniques include rotation, width and height shifts, shear transformations, zoom, and horizontal flips. This ensures the model generalizes well to various facial expressions in different orientations and lighting conditions.

## Training Process
The model is trained on a dataset of facial images with the following specifications:

Image Size: 48x48 pixels
Color Mode: Grayscale
Batch Size: 64
Epochs: 15
The dataset is split into training and validation sets, and data augmentation is applied to the training set only. The model's performance is monitored using a validation set, and a checkpoint is used to save the best model weights based on validation accuracy.

## GUI Implementation
A graphical user interface (GUI) is built using Tkinter, allowing users to upload an image and detect the emotion of the person in the image. The GUI includes the following features:

Image Upload: Users can upload an image from their file system.
Emotion Detection: The uploaded image is processed to detect faces, and the model predicts the emotion for each detected face.
Display Results: The predicted emotion is displayed on the GUI.
Required Libraries
TensorFlow
Keras
OpenCV
NumPy
PIL (Pillow)
Tkinter
Usage
