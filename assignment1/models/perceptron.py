"""Perceptron model."""

import numpy as np


class Perceptron:
    def __init__(self, n_class: int, lr: float, epochs: int):
        """Initialize a new classifier.

        Parameters:
            n_class: the number of classes
            lr: the learning rate
            epochs: the number of epochs to train for
        """
        # initialize weights to be all zeros, with 10 rows (for each class)
        # and X_train[1] number of columns
        self.w =  None # TODO: change this
        self.lr = lr
        self.epochs = epochs
        self.n_class = n_class

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train the classifier.

        Use the perceptron update rule as introduced in the Lecture.

        Parameters:
            X_train: a number array of shape (N, D) containing training data;
                N examples with D dimensions
            y_train: a numpy array of shape (N,) containing training labels
        """
        np.random.seed(0)
        self.w = np.random.rand(X_train.shape[1], self.n_class)

        for e in range(0, self.epochs):

            # iterate through each training sample
            for xi, yi in zip(X_train, y_train):
                wx = np.dot(xi, self.w) # (1 x D) * (D x 10)

                # loop through each class
                for c in range(0, self.n_class):
                    if wx[c] > wx[yi]:
                        self.w[:,yi] = self.w[:,yi] + self.lr * xi
                        self.w[:,c] = self.w[:,c] - self.lr * xi
        

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
