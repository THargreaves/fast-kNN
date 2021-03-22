"""A naive approach to kNN, running in O(mndk) time."""

from base import Base


class Standard(Base):

    def predict(self, X_test, K):
        """See Base.predict"""
        pred = []
        for x0 in X_test:
            neighbour_values = []
            for i, (x, y) in enumerate(zip(self.X_train, self.y_train)):
                d = self.distance(x0, x)
                neighbour_values.append((d, y))
            neighbour_values.sort(key=lambda e: e[0])
            pred.append(sum(e[1] for e in neighbour_values[:K]) / K)
        return pred
