# Handwritten-Character-Recognition-using-CNN
Recognizing handwritten character image using CNN with the CNN model trained using EMNIST dataset. EMNIST dataset is extended by adding 12 more characters from Tamil language to the dataset and prediction is made.
## Abstract
An attempt is made to recognize handwritten characters for English alphabets using multilayer Feed Forward neural network. EMNIST dataset which consists of English alphabets and numbers are made use of to train the neural network. EMNIST balanced dataset consist of  131,600 images of characters and 47 classes .The feature extraction technique is obtained by normalizing the pixel values. Pixel values will range from 0 to 255 which represents the intensity of each pixel in the image and they are normalized to represent value between 0 and 1. Convolutional neural network is used as a classifier which trains the EMNIST dataset. The work is extended by adding some more dataset to EMNIST dataset of characters from Tamil language and training the model. The prediction for the given input image is obtained from the trained classifier.

## Architecture
<img src="https://user-images.githubusercontent.com/26201632/39696619-e251f7dc-520b-11e8-9227-279ea40b4d6a.PNG" width="600">

## Packages used
- [Tensorflow](https://www.tensorflow.org/)
- [Scikit-learn](http://scikit-learn.org)
- [Tkinter](https://wiki.python.org/moin/TkInter)
- [Numpy](http://www.numpy.org/)
- [Matplotlib](https://matplotlib.org/)
- [Pillow](https://pillow.readthedocs.io/en/5.1.x/)
- [ipython](https://ipython.org/ipython-doc/2/install/install.html)

## Dataset
[EMNIST](https://www.nist.gov/itl/iad/image-group/emnist-dataset) dataset is downloaded and made use by using [python mnist-parser](https://github.com/sorki/python-mnist). Following python code will load emnist-balanced dataset to a python variable:
```
from mnist import MNIST
emnist = MNIST('C:\\Users\\Anandh\\Anaconda3\\final year project\emnist_data')
emnist.select_emnist('balanced')
images,labels = emnist.load_training()
testIM,testLAB = emnist.load_testing()
```
## Running the program
```
ipython filename.py
```
***note: the CNN model has been trained in emnistcnnmodel.ipynb and the model is saved and restored in the python file.***
## Screenshots

### Alphabets

<img src="https://user-images.githubusercontent.com/26201632/39698066-ddbbc09a-5210-11e8-9585-8fc620de0318.PNG" width="600">

### Numbers

<img src="https://user-images.githubusercontent.com/26201632/39698067-e2cbab40-5210-11e8-90cf-56d6d6917e99.PNG" width="600">

### Tamil Characters

<img src="https://user-images.githubusercontent.com/26201632/39698074-e718c250-5210-11e8-8058-9a92c6d9e8e4.PNG" width="600">

### Working

- Getting input image:

  <img src="https://user-images.githubusercontent.com/26201632/39697274-02a9c0da-520e-11e8-8a04-f58b32e9f70c.PNG" width="400">
  
  
- Displaying input image:

  <img src="https://user-images.githubusercontent.com/26201632/39697297-18a12ed2-520e-11e8-9f4a-a1578c856f3a.PNG" width="400">

- Rescaled Image(converting colored image to gray-scale and resizing to size of 28*28(EMNIST dataset size):

  <img src="https://user-images.githubusercontent.com/26201632/39697315-27bfccb6-520e-11e8-8a6a-97a044e23cf3.PNG" width="400">

- Normalized Image:

  <img src="https://user-images.githubusercontent.com/26201632/39697416-80fc1a50-520e-11e8-8022-bce9057dfcde.PNG" width="400">

- Predicted character:

  <img src="https://user-images.githubusercontent.com/26201632/39697439-90f7db88-520e-11e8-89f9-5d347340bc3e.PNG" width="400">

- Copied to clipboard:

  <img src="https://user-images.githubusercontent.com/26201632/39697442-94ce67b8-520e-11e8-83a4-f7d6b2137b95.PNG" width="400">

