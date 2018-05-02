'''
Artificial Neural Networks

Lei Mao

University of Chicago

Simplified Scikit-Learn Styled Multi-layer Perceptron Classifier Implemented From Scratch.

Experience:

Gradient Descent is slow and bull-shit.

Batch (update weights after seeing several training examples) Gradient Descent is good and fast.

'''

import numpy as np
import pandas as pd
import copy


def sigmoid(z):

    # Sigmoid function

    return 1.0 / (1.0 + np.exp(-z))

'''
def sigmoid(z):

    # Trick to prevent possible overflow of np.exp(-z)
    # Output will never to be exact 0
    idx = (z < -500)
    z[idx] = -500

    return 1.0 / (1.0 + np.exp(-z))
'''

def sigmoid_prime(z):

    # Derivative of the sigmoid function

    return sigmoid(z) * (1 - sigmoid(z))

def xavier_initializer(dim_in, dim_out):

    # Xavier initializer

    low = -np.sqrt(6. /(dim_in + dim_out))
    high = np.sqrt(6. /(dim_in + dim_out))

    return np.random.uniform(low = low, high = high, size = (dim_in, dim_out))


class OneHotEncoder():

    # Homebrew OneHotEncoder

    def fit(self, y):

        self.unique_classes = np.unique(y)
        self.n_classes = len(self.unique_classes)

        self.dictionary = dict(zip(self.unique_classes, list(range(self.n_classes))))
        self.dictionary_inverse = dict(zip(list(range(self.n_classes)), self.unique_classes))

    def fit_transform(self, y):

        self.unique_classes = np.unique(y)
        self.n_classes = len(self.unique_classes)

        self.dictionary = dict(zip(self.unique_classes, list(range(self.n_classes))))
        self.dictionary_inverse = dict(zip(list(range(self.n_classes)), self.unique_classes))

        onehot = np.zeros([len(y), self.n_classes])

        y_flattern = y.flattern()

        for i in range(len(y_flattern)):
            onehot[i,self.dictionary[y_flattern[i]]] = 1

        return onehot

    def transform(self, y):

        onehot = np.zeros([len(y), self.n_classes])

        y_flattern = y.flatten()

        for i in range(len(y_flattern)):
            onehot[i,self.dictionary[y_flattern[i]]] = 1

        return onehot

    def inverse_transform(self, y_onehot):

        indices = np.argmax(y_onehot, axis = 1)

        original_classes = list()
        for index in indices:
            original_classes.append(self.dictionary_inverse[index])
        original_classes = np.array(original_classes)

        return original_classes


class ANN(object):

    def __init__(self, h, s):

        # h: number of hidden layers
        # s: number of hidden units in each layer

        self.h = h
        self.s = s

        self.batch_size = 1

    def fit(self, X, y, alpha, t):

        self.enc = OneHotEncoder()
        self.enc.fit(y)
        y_onehot = self.enc.transform(y)
        n_samples, n_features = X.shape
        n_classes = y_onehot.shape[1]

        # Adaption for more flexible neural networks if necessary in the future
        self.sizes = [n_features] + [self.s for i in range(self.h)] + [n_classes]
                  
        self.biases = [np.random.randn(1, y) for y in self.sizes[1:]]
        self.weights = [xavier_initializer(dim_in = x, dim_out = y) for x, y in zip(self.sizes[:-1], self.sizes[1:])]

        self.training_loss = list()
        self.training_accuracy = list()

        for iteration in range(t):

            print("Interation: %d" %iteration)
            for i in range(n_samples // self.batch_size):

                start = i * self.batch_size
                end = (i + 1) * self.batch_size

                delta_biases, delta_weights, loss = self.gradient_descent(X[start:end,:], y_onehot[start:end,:])
                self.update_weights(delta_biases, delta_weights, alpha)

            training_loss = self.feedforward(X, y_onehot, self.weights, self.biases)
            training_accuracy = self.model_accuracy(X, y)
            self.training_loss.append(training_loss)
            self.training_accuracy.append(training_accuracy)

            print("Training Loss: %f" % training_loss)
            print("Training Accuracy: %f" % training_accuracy)

            
    def predict_prob(self, X):

        # Predict the probability of each class
        
        a = X
        for b, w in zip(self.biases, self.weights):
            z = np.dot(a, w) + b
            a = sigmoid(z)

        return a

    def predict_label(self, X):

        # Predict the most likely class
        
        a = X
        for b, w in zip(self.biases, self.weights):
            z = np.dot(a, w) + b
            a = sigmoid(z)

        predictions = self.enc.inverse_transform(a).reshape((-1,1))
        
        return predictions

    def model_accuracy(self, X, y):

        # Evaluate the trained model

        y_predict = self.predict_label(X)

        return np.sum(y_predict == y) / len(y)


    def update_weights(self, delta_biases, delta_weights, alpha):

        # Update weights using gradient

        assert (len(delta_biases) == len(self.biases))
        assert (len(delta_weights) == len(self.weights))

        for i in range(len(delta_biases)):
            self.biases[i] -= alpha * delta_biases[i]
        for i in range(len(delta_weights)):
            self.weights[i] -= alpha * delta_weights[i]

    def gradient_descent(self, X, y):

        # Gradient descent
        # Combination of forward propagation and back propagation

        hidden_values, activations, loss = self.fowardprop(X, y)
        delta_biases, delta_weights = self.backprop(X, y, hidden_values, activations)    

        return delta_biases, delta_weights, loss

    def fowardprop(self, X, y):

        # Foward propagation

        n_samples, n_features = X.shape

        activations = list()
        hidden_values = list()
        
        a = X
        activations.append(a)

        for b, w in zip(self.biases, self.weights):

            z = np.dot(a, w) + b
            a = sigmoid(z)
            hidden_values.append(z)
            activations.append(a)

        output = activations[-1]

        # Squre loss
        loss = 0.5 * np.sum((y - output) ** 2) / n_samples

        return hidden_values, activations, loss

    def backprop(self, X, y, hidden_values, activations):

        # Vectorized backpropagation for any batch sizes
        # Verified by gradient checking

        delta_biases = [np.zeros(b.shape) for b in self.biases]
        delta_weights = [np.zeros(w.shape) for w in self.weights]

        n_samples, n_features = X.shape
        n_layer = len(activations)

        output = activations[-1]
        hidden = hidden_values[-1]

        d_a = (output - y) / n_samples
        d_in = d_a * sigmoid(hidden) * (1 - sigmoid(hidden))
        delta_bias = np.sum(d_in, axis = 0, keepdims = True)
        delta_biases[-1] = delta_bias
        delta_weight = np.dot(activations[-2].T, d_in)
        delta_weights[-1] = delta_weight

        for l in range(2, n_layer):
            hidden = hidden_values[-l]
            d_a = np.dot(d_in, self.weights[-l+1].T)
            d_in = d_a * sigmoid(hidden) * (1 - sigmoid(hidden))
            delta_bias = np.sum(d_in, axis = 0, keepdims = True)
            delta_biases[-l] = delta_bias
            delta_weight = np.dot(activations[-l-1].T, d_in)
            delta_weights[-l] = delta_weight

        return delta_biases, delta_weights


    def feedforward(self, X, y, weights, biases):

        # Feedforward to calculate the loss using given X, y, weights and biases.

        n_samples, n_features = X.shape

        a = X
        for b, w in zip(biases, weights):
            z = np.dot(a, w) + b
            a = sigmoid(z)

        loss = 0.5 * np.sum((y - a) ** 2) / n_samples 

        return loss

    # Gradient checking suite for ANN

    def gradient_numerical(self, X, y, epsilon = 1):

        numgradelta_bias = [np.zeros(b.shape) for b in self.biases]
        numgradelta_weight = [np.zeros(w.shape) for w in self.weights]

        for w_idx in range(len(self.weights)):
            for (i, j), _ in np.ndenumerate(self.weights[w_idx]):
                weights_copy = copy.deepcopy(self.weights)
                biases_copy = copy.deepcopy(self.biases)
                weights_copy[w_idx][i,j] += epsilon
                loss_pos = self.feedforward(X = X, y = y, weights = weights_copy, biases = biases_copy)
                weights_copy = copy.deepcopy(self.weights)
                biases_copy = copy.deepcopy(self.biases)
                weights_copy[w_idx][i,j] -= epsilon
                loss_neg = self.feedforward(X = X, y = y, weights = weights_copy, biases = biases_copy)

                numgradelta_weight[w_idx][i,j] = (loss_pos - loss_neg) / (2 * epsilon)

        for b_idx in range(len(self.biases)):
            for (i, j), _ in np.ndenumerate(self.biases[b_idx]):
                weights_copy = copy.deepcopy(self.weights)
                biases_copy = copy.deepcopy(self.biases)
                biases_copy[b_idx][i,j] += epsilon
                loss_pos = self.feedforward(X = X, y = y, weights = weights_copy, biases = biases_copy)
                weights_copy = copy.deepcopy(self.weights)
                biases_copy = copy.deepcopy(self.biases)
                biases_copy[b_idx][i,j] -= epsilon
                loss_neg = self.feedforward(X = X, y = y, weights = weights_copy, biases = biases_copy)

                numgradelta_bias[b_idx][i,j] = (loss_pos - loss_neg) / (2 * epsilon)

        return numgradelta_bias, numgradelta_weight


    def gradient_check(self, X, y, epsilon = 1):

        self.enc = OneHotEncoder()
        self.enc.fit(y)
        y_onehot = self.enc.transform(y)
        n_samples, n_features = X.shape
        n_classes = y_onehot.shape[1]

        # Adaption for more flexible neural networks if necessary in the future
        self.sizes = [n_features] + [self.s for i in range(self.h)] + [n_classes]
                  
        self.biases = [np.random.randn(1, y) for y in self.sizes[1:]]
        self.weights = [xavier_initializer(dim_in = x, dim_out = y) for x, y in zip(self.sizes[:-1], self.sizes[1:])]


        delta_biases, delta_weights, loss = self.gradient_descent(X, y)

        numgradelta_biass, numgradelta_weights = self.gradient_numerical(X = X, y = y, epsilon = epsilon)

        diff_total = 0
        grad_total = 0
        n_parameters = 0

        for numgradelta_bias, delta_bias in zip(numgradelta_biass, delta_biases):
            diff_b = np.abs((numgradelta_bias - delta_bias))
            grad_total += np.sum(np.abs(delta_bias))
            diff_total += np.sum(diff_b)
            n_parameters += diff_b.shape[0] * diff_b.shape[1]
        for numgradelta_weight, delta_weight in zip(numgradelta_weights, delta_weights):
            diff_w = np.abs((numgradelta_weight - delta_weight))
            grad_total += np.sum(np.abs(delta_weight))
            diff_total += np.sum(diff_w)
            n_parameters += diff_w.shape[0] * diff_w.shape[1]
                
        diff_ave = diff_total / n_parameters
        grad_ave = grad_total / n_parameters

        return diff_ave, grad_ave

    def print(self):

        # Print weights and biases

        print(self.weights)
        print(self.biases)

    def get_parameters(self):

        # Get parameters:

        return {'weights': self.weights, 'biases': self.biases}

