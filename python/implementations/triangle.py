"""A naive approach to kNN, running in O(mndk) time."""

import math

from base import Base


class TriangleInequality(Base):

    def __init__(self, X_train, y_train, distance=None):
        super().__init__(X_train, y_train, distance)
        # Pre-computation of training distance matrix
        self.train_dist = [
            [
                math.sqrt(self.distance(self.X_train[i], self.X_train[j]))
                for j in range(i + 1, self.N)
            ]
            for i in range(self.N - 1)
        ]

    def predict(self, X_test, K):
        """See Base.predict"""
        pred = []
        for x0 in X_test:
            possible = [True for i in range(self.N)]
            curr_neighbours = [None for i in range(K)]

            for i, (x, y) in enumerate(zip(self.X_train, self.y_train)):
                if not possible[i]:
                    continue
                d = math.sqrt(self.distance(x0, x))
                if i >= K:
                    for j in range(self.N - (i + 1)):
                        if not possible[j]:
                            continue
                        if abs(d - self.train_dist[i][j]) >\
                                curr_neighbours[K - 1][3]:
                            possible[j + i + 1] = False
                for k in range(K):
                    if curr_neighbours[k] is None or \
                            curr_neighbours[k][3] > d:
                        curr_neighbours.insert(k, (i, x, y, d))
                        del curr_neighbours[K]
                        break

            pred.append(sum(e[2] for e in curr_neighbours) / K)
        return pred
