"""A naive implementation of the kNN algorithm."""

from base import Base


class Naive(Base):
    """A naive approach to kNN, running in O(mndk) time.

    Algorithm:
    1. For each test data point
        1. Initially no points are selected
        2. Loop through the training dataset to find the closest point that has
           not already been selected
        3. Add this point to the selections
        4. Repeat 1-3 k times
        5. Predict use the k nearest neighbours
    """

    def predict(self, X_test, K):
        """See Base.predict."""
        y_pred = []
        for x0 in X_test:
            selected = [False for __ in range(self.N)]
            neighbour_values = []
            for k in range(K):
                nearest = None  # (i, x, y, d)
                for i, (x, y) in enumerate(zip(self.X_train, self.y_train)):
                    if selected[i]:
                        continue
                    d = self.distance2(x0, x)
                    if nearest is None or d < nearest[3]:
                        nearest = i, x, y, d
                selected[nearest[0]] = True
                neighbour_values.append(nearest[2])
            y_pred.append(sum(neighbour_values) / K)
        return y_pred
