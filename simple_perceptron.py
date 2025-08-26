# A very simple perceptron for linear separable data 

import numpy as np

class Perceptron:
    def __init__(self, learning_rate=0.1, epochs=100):
        self.learning_rate = learning_rate
        self.epochos = epochs
    # epoch refers to one complete pass through the entire trainig dataset 


    # a method that instructts the algorithm to learn from the training data
    # X = input data 
    # no. of columns of weights = number of input data 
    # y = the true target/ label
    def fit(self, X, y):
        self.weights = np.random.rand(X.shape[1])
        # X.shape[0] gives number of rows in X
        # X.shape[1] gives number of columns in X 
        # generate a Numpy array with X.shape[1] length/columns that takes values between 0 and 1

        self.bias = np.random.rand(1)

        """
        extra info:
        To generate a random number in range [a,b]
        1. generate a random number between [0,1]
            n = np.random.rand(1)
        
        2. adjust the range 
            desired_n = (b-a) * n + a
        """

        for _ in range(self.epochos):
            for inputs, target in zip(X, y):
                # this step is just labelling the data/ preparing it for the rest of the function
                prediction = self.predict(inputs)
                error = target - prediction
                self.weights += self.learning_rate * error * inputs
                self.bias += self.learning_rate * error
                # for each weight and bias, nudge it a little bit to reduce the error


    def predict(self, inputs):
        weighted_sum = np.dot(inputs, self.weights) + self.bias
        return self._activation(weighted_sum)
        
    def _activation(self, z):
        # this is just a Heaviside step function so we get a binary output 
        return 1 if z >= 0 else 0



