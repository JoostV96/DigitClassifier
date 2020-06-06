# DigitClassifier
Classifier to identify your own handwritten digits using a GUI. We train a convolutional neural network on the MNIST handwritten digits and use this model to classify our own handwritten digits. 
It consits of 2 Python files and the trained model:

* MNIST_model.py: trains the convolutional neural network and saves the model as `MNIST_model.h5`. The trained model is added in case one wants to skip the training part.
* DrawNumber.py: provides the user with a canvas on which they can draw their own digit. This file loads the above mentioned trained model. After the user pressed "DONE", the drawing is used as input for the model to make a prediction. This prediction is then given to the user on a pop-up window. 
