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
            possible = [i for i in range(self.N)]
            curr_neighbours = [None for i in range(K)]

            i = 0
            n = self.N
            while i < n:
                p = possible[i]
                x = self.X_train[p]
                y = self.y_train[p]
                d = math.sqrt(self.distance(x0, x))
                if i >= K:
                    j = i + 1
                    while j < n:
                        if abs(d - self.train_dist[p][possible[j] - (p + 1)]) >\
                                curr_neighbours[K - 1][3]:
                            del possible[j]
                            n -= 1
                        else:
                            j += 1
                for k in range(K):
                    if curr_neighbours[k] is None or \
                            curr_neighbours[k][3] > d:
                        curr_neighbours.insert(k, (i, x, y, d))
                        del curr_neighbours[K]
                        break
                i += 1

            pred.append(sum(e[2] for e in curr_neighbours) / K)
        return pred
