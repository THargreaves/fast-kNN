"""An abstract base class for other kNN implementations."""

from abc import ABC, abstractmethod


class Base:

    def __init__(self, X_train, y_train):
        """Instantiate a kNN regressor with training data.

        :param X_train: Training data predictors
        :param y_train: Training data labels
        """
        self.X_train = X_train
        self.y_train = y_train

        self.N = len(X_train)
        self.P = len(X_train[0])

    @staticmethod
    def distance2(x1, x2):
        """Squared Euclidean distance between two points.

        :param x1: The first data point
        :param x2: The second data point
        :return d2: Squared Euclidean distance
        """
        return sum((e1 - e2) ** 2 for e1, e2 in zip(x1, x2))

    @abstractmethod
    def predict(self, X_test, k):
        """Predict response for new data points.

        :param X_test: A 2D list-of-lists of new data points
        :param k: Number of neighbours to use
        :return pred: A list of predicted values
        """
        pass
