"""An implementation of the kNN algorithm based on the triangle inequality."""

import math

from .base import Base


class TriangleInequality(Base):
    """A triangle-inequality-based approach to kNN.

    Algorithm:
    1. For each test data point
        1. Initially mark all training data points as possibly a neighbour
        2. Create an empty list to store the candidates for the k neighbours
        3. For each training data point that is still possibly a neighbour
            1. Calculate the distance to the test point
            2. Compare to existing neighbour candidates and add if appropriate
            3. If added to candidates, use triangle inequality checks to remove
               training data points that can no longer be a neighbour
        4. Predict use the k nearest neighbours
    """

    def __init__(self, X_train, y_train):
        """See Base.__init__."""
        super().__init__(X_train, y_train)
        # Pre-computation of training distance matrix
        self._train_dist = [
            [
                math.sqrt(self.distance2(self.X_train[i], self.X_train[j]))
                for j in range(i + 1, self.N)
            ]
            for i in range(self.N - 1)
        ]

    def train_dist(self, i, j):
        """Getter method for training distance matrix.

        Since distance matrices are symmetric, they can be stored more
        efficiently as a ragged array. This getter method is used to translate
        matrix indexes into the ragged indexes for the internal representation.

        :param i: Index of the first training data point
        :param j: Index of the second training data point
        :return d: Euclidean distance between training points i and j
        """
        if i == j:
            return 0
        if j > i:
            i, j = j, i
        return self._train_dist[i][j - (i + 1)]

    def predict(self, X_test, K):
        """See Base.predict"""
        y_pred = []
        for x0 in X_test:
            possible = [i for i in range(self.N)]
            curr_neighbours = [None for __ in range(K)]  # (i, x, y, d)

            n1 = 0  # number of data points already considered
            n = self.N  # number of possible data points remaining
            while n1 < n:
                i = possible[n1]
                x = self.X_train[i]
                y = self.y_train[i]
                d = math.sqrt(self.distance2(x0, x))

                # Compare to current k-nearest neighbours
                for k in range(K):
                    # Check if a new candidate for neighbour
                    if curr_neighbours[k] is None or curr_neighbours[k][3] > d:
                        curr_neighbours.insert(k, (n1, x, y, d))
                        del curr_neighbours[K]

                        if k < K:
                            break

                        # Apply triangle inequality to remove existing points
                        n2 = n1 + 1
                        while n2 < n:
                            j = possible[n2]
                            d_tri = abs(d - self.train_dist(i, j))
                            if d_tri > curr_neighbours[K - 1][3]:
                                del possible[n2]
                                n -= 1
                            else:
                                n2 += 1

                        break
                n1 += 1

            y_pred.append(sum(e[2] for e in curr_neighbours) / K)
        return y_pred
