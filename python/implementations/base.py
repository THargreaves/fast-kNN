"""An abstract base class for other kNN implementations."""

from abc import ABC, abstractmethod


class Base:

    def __init__(self, X_train, y_train, distance=None):
        self.X_train = X_train
        self.y_train = y_train
        if not distance:
            def distance(x1, x2):
                return sum((e1 - e2) ** 2 for e1, e2 in zip(x1, x2))
        self.distance = distance

        self.N = len(X_train)
        self.P = len(X_train[0])

    @abstractmethod
    def predict(self, X_test, k):
        """Predict response for new data points.

        :param X_test: A 2D list-of-lists of new data points
        :param k: Number of neighbours to use
        :return pred: A list of predicted values
        """
        pass
