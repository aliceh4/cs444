"""Softmax model."""

import numpy as np


class Softmax:
    def __init__(self, n_class: int, lr: float, epochs: int, reg_const: float):
        """Initialize a new classifier.

        Parameters:
            n_class: the number of classes
            lr: the learning rate
            epochs: the number of epochs to train for
            reg_const: the regularization constant
        """
        self.w = None  # TODO: change this
        self.lr = lr
        self.epochs = epochs
        self.reg_const = reg_const
        self.n_class = n_class
        
    def calc_gradient(self, X_train: np.ndarray, y_train: np.ndarray) -> np.ndarray:
        """Calculate gradient of the softmax loss.

        Inputs have dimension D, there are C classes, and we operate on
        mini-batches of N examples.

        Parameters:
            X_train: a numpy array of shape (N, D) containing a mini-batch
                of data
            y_train: a numpy array of shape (N,) containing training labels;
                y[i] = c means that X[i] has label c, where 0 <= c < C

        Returns:
            gradient with respect to weights w; an array of same shape as w
        """
        gradient = np.zeros(self.w.shape) # set gradient equal to weight*constant
        for xi, yi in zip(X_train, y_train):
            wx = np.dot(xi, self.w) # (1, D) * (D * n_class) = 1 x n_class array
            max = np.max(wx)
            wx_exp = np.exp(wx - max)
            sum = np.sum(wx_exp)
            gradient[:,yi] += -xi + wx_exp[yi] * xi / (sum)


            for c in range(0, self.n_class):
                if c != yi:
                    gradient[:, c] += wx_exp[c] * xi / (sum)

        return gradient

    def create_mini_batches(self, X, y, batch_size):
        mini_batches = []
        n_minibatches = int(X.shape[0] / batch_size)
    
        for i in range(0, n_minibatches):
            X_mini = X[i * batch_size: (i+1) * batch_size]
            Y_mini = y[i * batch_size: (i+1) * batch_size]
            mini_batches.append((X_mini, Y_mini))
        return mini_batches

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train the classifier.

        Hint: operate on mini-batches of data for SGD.

        Parameters:
            X_train: a numpy array of shape (N, D) containing training data;
                N examples with D dimensions
            y_train: a numpy array of shape (N,) containing training labels
        """
        batch_size = 250
        mini_batches = self.create_mini_batches(X_train, y_train, batch_size)

        np.random.seed(0)
        self.w = np.random.rand(X_train.shape[1], self.n_class) # (D x n_class) matrix
        for e in range(0, self.epochs):
            if e % 10 == 1:
                # decay
                self.lr = self.lr / 2
            for batch in mini_batches:
                x_mini, y_mini = batch
                gradient = self.calc_gradient(x_mini, y_mini) / batch_size
                self.w += -self.lr * gradient
                    

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Use the trained weights to predict labels for test data points.

        Parameters:
            X_test: a numpy array of shape (N, D) containing testing data;
                N examples with D dimensions

        Returns:
            predicted labels for the data in X_test; a 1-dimensional array of
                length N, where each element is an integer giving the predicted
                class.
        """
        N = X_test.shape[0]
        predicted = np.zeros(N)
        for i in range(0, N): 
            xi = X_test[i]
            wx = np.dot(xi.T, self.w) # (1, D) * (D x 10)
            predicted[i] = np.argmax(wx)
        return predicted.astype(int)
