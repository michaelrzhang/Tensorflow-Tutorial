# MNIST Exercise

This exercise is intended for those with a decent understanding of neural networks who want to try using Tensorflow without copy pasting a lot of code. The idea is that looking up functions is more helpful for learning a new framework.

## Directions

1. Load data. One way to do this is via this link: https://pypi.python.org/pypi/python-mnist/. 

`pip install mnist`

Separate the training data into 50,000 points for training and 10,000 for validation.

2. Set up your network inputs. You should look up how placeholders work. What is the shape of your data?

3. Specify some variables to carry out your computation. To keep your architecture simple, we recommend using just one layer of weights and biases. You will find tf.nn.matmul helpful. 

4. Use the softmax function to convert the output of your previous step to a probability distribution. Use mean squared loss to compute your error.

5. Set up an optimizer to minimize your above loss function. You can use tf.train.GradientDescentOptimizer.

6. Initialize your variables within a session and start running your optimizer!

7. After training for a while (or something like every 1000 iterations), print out the accuracy of your classifier on the validation data. You will find tf.argmax to be useful.

8. Improve your neural net architecture. Look up how to use convolutional layers, add more layers, use more sophisicated optimizers, and train for longer!

