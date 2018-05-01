from ann import ANN 
from utils import plot_curve

import time
import numpy as np
import pandas as pd

# Scikit-Learning only used for datasets and data preparation
from sklearn import datasets
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

def iris_gradient_test():

    # Gradient checking using Iris dataset

    print("Gradient checking using Iris dataset")

    iris = datasets.load_iris()

    X = iris.data
    y = iris.target
    X, y = shuffle(X, y, random_state = 0)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)
    y_train = y_train.reshape((-1,1))
    y_test = y_test.reshape((-1,1))

    nn = ANN(h = 2, s = 64)
    abs_diff_ave, abs_grad_ave = nn.gradient_check(X_train, y_train, epsilon = 1e-5)

    print("Averaged Absolute Difference between Gradient Descent and Numerical Gradient Approximate: %s" % str(abs_diff_ave))
    print("Averaged Absolute Value of Gradient Calculated by Gradient Descent: %s" % str(abs_grad_ave))
    print("Difference Ratio: %s" % str(abs_diff_ave / abs_grad_ave))

    if abs_diff_ave / abs_grad_ave < 1e-4:
        print("Gradient Checking Passed.")
    else:
        print("Gradient Checking Failed!")


def iris():

    # Test ANN on Iris dataset

    iris = datasets.load_iris()

    X = iris.data
    y = iris.target
    X, y = shuffle(X, y, random_state = 0)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)
    y_train = y_train.reshape((-1,1))
    y_test = y_test.reshape((-1,1))

    nn = ANN(h = 2, s = 64)
    nn.fit(X_train, y_train, alpha = 0.2, t = 100)

    print("Train Accuracy:", nn.model_accuracy(X_train, y_train))
    print("Test Accuracy:", nn.model_accuracy(X_test, y_test))

    #parameters = nn.get_parameters()
    #print("Weights:")
    #print(parameters['weights'])
    #print("Biases:")
    #print(parameters['biases'])

def mnist():

    print("Train MNIST dataset using ANN")

    # Test ANN on MNIST dataset

    data = pd.read_csv('data/train.csv')
    y = data['label'].as_matrix().reshape((-1,1))
    X = data.iloc[:,1:].as_matrix() / 255
    X, y = shuffle(X, y, random_state = 0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    y_train = y_train.reshape((-1,1))
    y_test = y_test.reshape((-1,1))

    # Some good parameters
    # h = 1, s = 128, alpha = 1,t = 10
    # h = 1, s = 64, alpha = 1, t = 10
    # h = 2, s = 64, alpha = 5, t = 30 # Had a long stationary stage before actually see loss decrease
    # h = 2, s = 64, alpha = 1, t = 30 # Train Accuracy: 0.954404761905 Test Accuracy: 0.938214285714

    nn = ANN(h = 2, s = 128)
    nn.fit(X_train, y_train, alpha = 0.1, t = 10)

    print("Train Accuracy:", nn.model_accuracy(X_train, y_train))
    print("Test Accuracy:", nn.model_accuracy(X_test, y_test))

    plot_curve(losses = nn.training_loss, accuracies = nn.training_accuracy, savefig = True, showfig = False, filename = 'training_curve.png')

    #parameters = nn.get_parameters()
    #print("Weights:")
    #print(parameters['weights'])
    #print("Biases:")
    #print(parameters['biases'])


if __name__ == '__main__':

    start_time = time.time()

    np.random.seed(0)
    iris_gradient_test()

    print("")

    #np.random.seed(0)
    #iris()
    #print("")

    np.random.seed(0)
    mnist()
    end_time = time.time()

    time_elapsed = end_time - start_time

    print("Time Elapsed: %02d:%02d:%02d" % (time_elapsed // 3600, (time_elapsed % 3600 // 60), (time_elapsed % 60 // 1)))


