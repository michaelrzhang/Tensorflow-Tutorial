# MNIST Exercise

This exercise is intended for those with a decent understanding of neural networks who want to try using Tensorflow without copy pasting a lot of code. Feel free to refer to documentations and examples online, but try to do most of the actual coding yourself. The idea is that looking up functions is more helpful for learning a new framework. We recommend using jupyter notebook rather than vanilla python.

## Resource Overview
Tensorflow is a python module that lets you specify the structure of your neural net (properties and connections of the different layers and such), specify an optimization algorithm (e.g. GradientDescent), and then lets you train the net by passing data to it repeatedly.

#### Installing Tensorflow on Windows
1. https://www.tensorflow.org/install/install_windows
2. NOTE:
    * Get Anaconda if you didn’t
    * use this command to create your tensorflow environment: conda create --name tensorflow python=3.5
    * Let Michael or Zee know if you have issue with installation. Zee confirmed Windows tensorflow works

MNIST is a commonly used beginner dataset. You get a training set of X and label, and a testing X. The X's are a matrix of dimension n x 784, with each row representing a flattened digit image (28 x 28). The labels are a matrix of dimension n x 1, each row being the digit 0 through 9.

## Directions

1. Load data. One way to do this is via this link: https://pypi.python.org/pypi/python-mnist/. Use the data in the data/ directory
```
pip install mnist
```
```
from mnist import MNIST

def load_dataset():
    mndata = MNIST('./data/')
    X_train, labels_train = map(np.array, mndata.load_training())
    # The test labels are meaningless
    # since you're replacing the official MNIST test set with our own test set
    X_test, _ = map(np.array, mndata.load_testing())
    return X_train, labels_train, X_test

X_train, labels_train, X_test = load_dataset()
```

Do not use `from tensorflow.examples.tutorials.mnist import input_data`

Separate the training set into 50,000 points for training and 10,000 for validation. Remember you need to preprocess the data: center and normalize the X's and one-hot encode the n x 1 label matrix into a n x 10 matrix.

2. Set up your network inputs. You should look up how placeholders work. What is the shape of your data? 

3. Specify some variables to carry out your computation. To keep your architecture simple, we recommend using just one layer of weights and biases. You will find tf.nn.matmul helpful. 

4. Use the softmax function to convert the output of your previous step to a probability distribution. Use the cross_entropy loss function on this probability distribution. You can try without regularization first, later you could try adding a regularization term to your loss.

5. Set up an optimizer to minimize your above loss function. You can use tf.train.GradientDescentOptimizer. Play with the learning rate here.

6. Initialize your variables within a session and start running your optimizer! Start with a batch size of 1 to get stuff working, but you can play around with batch size. Plot the loss with matplotlib every ~1000 samples.

7. After training for a while (or something like every 1000 iterations), print out the accuracy of your classifier on the validation data. You will find tf.argmax to be useful.

8. Improve your neural net architecture to increase your validation accuracy. Look up how to use convolutional layers, add more layers, use more sophisicated optimizers, and train for longer! Try adding dropout, changing layer sizes, adding regularization, or changing learning rate.

9. Use your net to predict the test set. Save the predictions to a csv with columns index, prediction (not one-hot encoded).

