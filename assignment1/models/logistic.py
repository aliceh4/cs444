"""Logistic regression model."""

import numpy as np
from pytz import NonExistentTimeError


class Logistic:
    def __init__(self, lr: float, epochs: int, threshold: float):
        """Initialize a new classifier.

        Parameters:
            lr: the learning rate
            epochs: the number of epochs to train for
        """
        self.w = None  # TODO: change this
        self.lr = lr
        self.epochs = epochs
        self.threshold = threshold

    def sigmoid(self, z: np.ndarray) -> np.ndarray:
        """Sigmoid function.

        Parameters:
            z: the input

        Returns:
            the sigmoid of the input
        """
        sigmoid = 1 / (1 + np.exp(-z))
        return sigmoid

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train the classifier.

        Use the logistic regression update rule as introduced in lecture.

        Parameters:
            X_train: a numpy array of shape (N, D) containing training data;
                N examples with D dimensions
            y_train: a numpy array of shape (N,) containing training labels
        """
        np.random.seed(0)
        self.w = np.random.rand(X_train.shape[1]) # (D x 1) matrix
        for _ in range(0, self.epochs):
            for xi, yi in zip(X_train, y_train):
                if yi == 0:
                    yi = -1
                
                wx = np.dot(xi, self.w) # (1 x D) * (D x 1) = (1 x 1)
                self.w = self.w + self.lr * self.sigmoid(-yi * wx) * yi * xi
                
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
        i = 0
        for xi in X_test:
            predicted_prob = np.dot(xi, self.w) # (1 x 22) * (22 x 1)
            if predicted_prob > 0:
                predicted[i] = 1
            else:
                predicted[i] = 0
            i += 1
        return predicted

