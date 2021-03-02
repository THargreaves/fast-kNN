"""A naive approach to kNN, running in O(mndk) time."""

from base import Base


class Naive(Base):

    def predict(self, X_test, K):
        """See Base.predict"""
        pred = []
        for x0 in X_test:
            selected = [False for i in range(self.N)]
            neighbour_values = []
            for k in range(K):
                nearest = None  # (i, X, y, d)
                for i, (x, y) in enumerate(zip(self.X_train, self.y_train)):
                    if selected[i]:
                        continue
                    d = self.distance(x0, x)
                    if nearest is None or d < nearest[2]:
                        nearest = i, x, y, d
                selected[i] = True
                neighbour_values.append(y)
            pred.append(sum(neighbour_values) / K)
        return pred
