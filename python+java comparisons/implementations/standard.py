"""A standard implementation of the kNN algorithm."""

from .base import Base


class Standard(Base):
    """A standard approach to kNN, running in O(mn(d + log(n))) time.

    Algorithm:
    1. For each test data point
        1. Compute the distance to each point in the training set
        2. Sort training points by these distances
        3. Select the points with the k smallest distances
        4. Predict use the k nearest neighbours
    """

    def predict(self, X_test, K):
        """See Base.predict."""
        y_pred = []
        for x0 in X_test:
            neighbour_values = []
            for i, (x, y) in enumerate(zip(self.X_train, self.y_train)):
                d = self.distance2(x0, x)
                neighbour_values.append((d, y))
            neighbour_values.sort(key=lambda e: e[0])
            y_pred.append(sum(e[1] for e in neighbour_values[:K]) / K)
        return y_pred
