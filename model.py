import random
import numpy as np
import pandas as pd
from challenge_basic import get_data, get_data2
import sys

import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB, CategoricalNB
from sklearn.exceptions import DataConversionWarning
import warnings




x_train, t_train, x_test, t_test = None, None, None, None

def warn(*args, **kwargs): # To suppress warnings
    pass


def gen_input_output():
    """
    Generate input-output pairs from a sequence of data.

    Parameters:
        `data` - a list of integers representing a sequence of notes

    Returns: a list of pairs of the form (x, t) where
        `x` - a numpy array of shape (20, 128) representing the input
        `t` - an integer representing the target note
    """
    global x_train, t_train, x_test, t_test
    file_name = "/Users/adityaharish/Documents/Documents/Subjects/University/UTM/Year_3/Winter_2024_Courses/CSC311/ML_Challenge/clean_dataset.csv"
    x_train, t_train, x_test, t_test = get_data2(file_name)

def softmax(z):
    """
    Compute the softmax of vector z, or row-wise for a matrix z.
    For numerical stability, subtract the maximum logit value from each
    row prior to exponentiation (see above).

    Parameters:
        `z` - a numpy array of shape (K,) or (N, K)

    Returns: a numpy array with the same shape as `z`, with the softmax
        activation applied to each row of `z`

    Given m=max_l*z_l compute: y_k=e^(z_k−m)/[∑_l(e(z_l−m))]
    """
    z2 = z.T
    m = np.max(z2, axis=0)
    diff = np.float32(z2 - m)
    y = np.exp(diff) / np.sum(np.exp(diff), axis=0)
    return y.T

class MLPModel(object):
    """
    Input for init = 

    layer_data has num_features=128*20, num_hidden=100, num_classes=128 for each layer

    [(num_features, num_classes)]
    """
    def __init__(self, layer_data=[(128*20, 100), (100, 128)]):
        """
        Initialize the weights and biases of this two-layer MLP.
        """
        # information about the model architecture
        self.num_features = layer_data[0][0]
        self.num_hidden = []
        for i in range(1, len(layer_data)):
            self.num_hidden.append(layer_data[i][0])

        self.num_classes = layer_data[-1][1]

        # weights and biases for the first layer of the MLP
        # note that by convention, the weight matrix shape has the
        # num input features along the first axis, and num output
        # features along the second axis
        # self.W1 = np.zeros([num_features, num_hidden])
        # self.b1 = np.zeros([num_hidden])

        # # weights and biases for the second layer of the MLP
        # self.W2 = np.zeros([num_hidden, num_classes])
        # self.b2 = np.zeros([num_classes])

        self.W, self.b = [], []
        for i in range(len(layer_data)):
            self.W.append(np.zeros([layer_data[i][0], layer_data[i][1]]))
            self.b.append(np.zeros([layer_data[i][1]]))

        # initialize the weights and biases
        self.initializeParams()

        # set all values of intermediate variables (to be used in the
        # forward/backward passes) to None
        self.cleanup()

    def initializeParams(self):
        """
        Initialize the weights and biases of this two-layer MLP to be random.
        This random initialization is necessary to break the symmetry in the
        gradient descent update for our hidden weights and biases. If all our
        weights were initialized to the same value, then their gradients will
        all be the same!
        """
        self.W[0] = np.random.normal(0, 2/self.num_features, self.W[0].shape)
        self.b[0] = np.random.normal(0, 2/self.num_features, self.b[0].shape)

        for i in range(1, len(self.W)):
            self.W[i] = np.random.normal(0, 2/self.num_hidden[i-1], self.W[i].shape)
            self.b[i] = np.random.normal(0, 2/self.num_hidden[i-1], self.b[i].shape)

    def forward(self, X):
        """
        Compute the forward pass to produce prediction for inputs.

        Parameters:
            `X` - A numpy array of shape (N, self.num_features)

        Returns: A numpy array of predictions of shape (N, self.num_classes)
        """
        return do_forward_pass(self, X) # To be implemented below

    def backward(self, ts):
        """
        Compute the backward pass, given the ground-truth, one-hot targets.

        You may assume that the `forward()` method has been called for the
        corresponding input `X`, so that the quantities computed in the
        `forward()` method is accessible.

        Parameters:
            `ts` - A numpy array of shape (N, self.num_classes)
        """
        return do_backward_pass(self, ts) # To be implemented below

    def loss(self, ts):
        """
        Compute the average cross-entropy loss, given the ground-truth, one-hot targets.

        You may assume that the `forward()` method has been called for the
        corresponding input `X`, so that the quantities computed in the
        `forward()` method is accessible.

        Parameters:
            `ts` - A numpy array of shape (N, self.num_classes)
        """
        return np.sum(-ts * np.log(self.y)) / ts.shape[0]

    def update(self, alpha):
        """
        Compute the gradient descent update for the parameters of this model.

        Parameters:
            `alpha` - A number representing the learning rate
        """
        for i in range(len(self.W) - 1):
            self.W[i] = self.W[i] - alpha * self.W_bar[i]
            self.b[i] = self.b[i] - alpha * self.b_bar[i]

    def cleanup(self):
        """
        Erase the values of the variables that we use in our computation.
        """
        # To be filled in during the forward pass
        self.N = None # Number of data points in the batch
        self.X = None # The input matrix
        self.m = [None] * (len(self.W))
        self.h = [None] * (len(self.W))

        self.z = None # The logit scores (pre-activation output values)
        self.y = None # Class probabilities (post-activation)
        # To be filled in during the backward pass
        self.z_bar = None # The error signal for self.z2
        self.W_bar = [None] * (len(self.W))
        self.b_bar = [None] * (len(self.W) - 1)
        self.h_bar = [None] * (len(self.W) - 1)
        self.m_bar = [None] * (len(self.W) - 1)

    def predict(self, X):
        """
        Make predictions for the given input data.

        Parameters:
            `X` - A numpy array of shape (N, self.num_features)

        Returns: A numpy array of predictions of shape (N, self.num_classes)
        """
        self.forward(X)
        index = np.argmax(self.y)
        return index
    
    def accuracy(self, X, t):
        """
        Compute the accuracy of the model given input data and target labels.
        """
        y = np.array([self.predict(x) for x in X])
        print(y)
        print(t)
        z = np.argmax(t, axis=1)
        print(z)
        return np.sum(y == z) / t.shape[0]
        
        

def do_forward_pass(model, X):
    """
    Compute the forward pass to produce prediction for inputs.

    This function also keeps some of the intermediate values in
    the neural network computation, to make computing gradients easier.

    For the ReLU activation, you may find the function `np.maximum` helpful

    Parameters:
        `model` - An instance of the class MLPModel
        `X` - A numpy array of shape (N, model.num_features)

    Returns: A numpy array of predictions of shape (N, model.num_classes)
    """
    model.N = X.shape[0]
    model.X = X
    temp_x = X
    for i in range(len(model.W) - 1):
        model.m[i] = (temp_x@model.W[i]) + model.b[i]
        model.h[i] = np.maximum(0, model.m[i])
        temp_x = model.h[i]
    model.z = (model.h[len(model.W) - 2]@model.W[len(model.W) - 1]) + model.b[len(model.W) - 1]
    model.y = softmax(model.z)
    # model.m = (X@model.W1) + model.b1 # TODO - the hidden state value (pre-activation)
    # model.h = np.maximum(0, model.m) # TODO - the hidden state value (post ReLU activation)
    # model.z = (model.h@model.W2) + model.b2 # TODO - the logit scores (pre-activation)
    # model.y = softmax(model.z) # TODO - the class probabilities (post-activation)
    return model.y

def do_backward_pass(model, ts):
    """
    Compute the backward pass, given the ground-truth, one-hot targets.

    You may assume that `model.forward()` has been called for the
    corresponding input `X`, so that the quantities computed in the
    `forward()` method is accessible.

    The member variables you store here will be used in the `update()`
    method. Check that the shapes match what you wrote in Part 2.

    Parameters:
        `model` - An instance of the class MLPModel
        `ts` - A numpy array of shape (N, model.num_classes)
    """
    model.z_bar = (model.y - ts) / model.N
    print(model.y[0:10])
    print(ts[0:10])
    
    model.W_bar[-1] = model.h[-2].T.dot(model.z_bar)
    model.b_bar[-1] = np.sum(model.z_bar, axis=0)

    for i in range(len(model.W) - 2, 0, -1):
        model.h_bar[i] = model.z_bar.dot(model.W[i + 1].T)
        model.m_bar[i] = model.h_bar[i] * (model.h[i] > 0)
        model.W_bar[i] = model.h[i - 1].T.dot(model.m_bar[i])
        model.b_bar[i] = np.sum(model.m_bar[i], axis=0)

    # Handle the first layer separately
    model.h_bar[0] = model.m_bar[1].dot(model.W[1].T)
    model.m_bar[0] = model.h_bar[0] * (model.h[0] > 0)
    model.W_bar[0] = model.X.T.dot(model.m_bar[0])
    model.b_bar[0] = np.sum(model.m_bar[0], axis=0)

def train_sgd(model, X_train, t_train,
              alpha=0.1, n_epochs=0, batch_size=100,
              X_valid=None, t_valid=None,
              w_init=None, plot=True):
    '''
    Given `model` - an instance of MLPModel
          `X_train` - the data matrix to use for training
          `t_train` - the target vector to use for training
          `alpha` - the learning rate.
                    From our experiments, it appears that a larger learning rate
                    is appropriate for this task.
          `n_epochs` - the number of **epochs** of gradient descent to run
          `batch_size` - the size of each mini batch
          `X_valid` - the data matrix to use for validation (optional)
          `t_valid` - the target vector to use for validation (optional)
          `w_init` - the initial `w` vector (if `None`, use a vector of all zeros)
          `plot` - whether to track statistics and plot the training curve

    Solves for model weights via stochastic gradient descent,
    using the provided batch_size.

    Return weights after `niter` iterations.
    '''
    # as before, initialize all the weights to zeros
    w = np.zeros(X_train.shape[1])

    train_loss = [] # for the current minibatch, tracked once per iteration
    valid_loss = [] # for the entire validation data set, tracked once per epoch

    # track the number of iterations
    niter = 0

    # we will use these indices to help shuffle X_train
    N = X_train.shape[0] # number of training data points
    indices = list(range(N))

    for e in range(n_epochs):
        random.shuffle(indices) # for creating new minibatches

        for i in range(0, N, batch_size):
            if (i + batch_size) > N:
                # At the very end of an epoch, if there are not enough
                # data points to form an entire batch, then skip this batch
                continue

            indices_in_batch = indices[i: i+batch_size]
            X_minibatch = X_train[indices_in_batch, :]
            t_minibatch = t_train[indices_in_batch]

            # gradient descent iteration
            model.cleanup()
            model.forward(X_minibatch)
            model.backward(t_minibatch)
            model.update(alpha)
            niter += 1


def run_logistic_regression():
    """
    Run logistic regression on the given dataset.
    """
    if any([x is None for x in [x_train, t_train, x_test, t_test]]):
        gen_input_output()

    # Create a logistic regression model with sklearn
    print("Running Logistic Regression")
    lr = LogisticRegression(verbose=0)
    lr.fit(x_train, t_train)

    print(f"Training Score:  {lr.score(x_train, t_train)}")
    print(f"Testing Score: {lr.score(x_test, t_test)}")

    print("-------------------\n\n")


def run_knn():
    """
    Run k-nearest neighbors on the given dataset. with different set of k values
    """
    if any([x is None for x in [x_train, t_train, x_test, t_test]]):
        gen_input_output()

    # Create a logistic regression model with sklearn
    n = x_train.shape[0]
    max_k = int(n ** 0.5)

    print("Running KNN")
    for i in range(1, max_k + 1, 2):
        knn = KNeighborsClassifier(n_neighbors=i)

        knn.fit(x_train, t_train)

        print("K = " + str(i) + ":")
        print(f"Training Score: {knn.score(x_train, t_train)}")
        print(f"Testing Score: {knn.score(x_test, t_test)}")
        print("-------------------\n\n")
    


def run_decision_tree():
    """
    Run the decision tree on the given dataset with different set of max_depth values
    """
    if any([x is None for x in [x_train, t_train, x_test, t_test]]):
        gen_input_output()


    # Create a logistic regression model with sklearn
    
    print("Running Decision Tree")

    criterion = ["gini", "entropy", 'log_loss']


    for crit in criterion:
        print("Criterion = " + crit)
        for i in range(1, 11): # 73 is the number of features we have
            dt = DecisionTreeClassifier(max_depth=i, criterion=crit)

            dt.fit(x_train, t_train)

            print("Max Depth = " + str(i) + ":")

            print(f"Training Score: {dt.score(x_train, t_train)}")
            print(f"Testing Score: {dt.score(x_test, t_test)}")

            print("-------------------\n\n")
        print("---------------------------\n\n")


def run_naive_bayes():
    """
    Run Naive Bayes on the given dataset.
    """

    if any([x is None for x in [x_train, t_train, x_test, t_test]]):
        gen_input_output()

    print("Running Naive Bayes")

    print("Gaussian Naive Bayes")
    nb = GaussianNB()

    nb.fit(x_train, t_train)

    print(f"Training Score: {nb.score(x_train, t_train)}")

    print(f"Testing Score: {nb.score(x_test, t_test)}")

    print("-------------------\n\n")

    print("Categorical Naive Bayes")





    







if __name__ == "__main__":
    warnings.warn = warn
    if len(sys.argv) != 2:
        print("Usage: python model.py <model_name>")
        sys.exit(1)
    if sys.argv[1] == "lr":
        run_logistic_regression()
    elif sys.argv[1] == "knn":
        run_knn()
    elif sys.argv[1] == "dt":
        run_decision_tree()
    elif sys.argv[1] == "nb":
        run_naive_bayes()
    elif sys.argv[1] == "all":
        print("Running all models")
        run_logistic_regression()
        run_knn()
        run_decision_tree()
        run_naive_bayes()

    else:
        print("Invalid model name. Choose from lr, knn, dt, nb, all")
        sys.exit(1)

    
            









