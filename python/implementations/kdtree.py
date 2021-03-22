"""A naive approach to kNN, running in O(mndk) time."""

import math

from base import Base


class KDTree(Base):

    def __init__(self, X_train, y_train, distance=None):
        super().__init__(X_train, y_train, distance)
        # Pre-computation of KD tree
        self.root = self._grow_tree(self.X_train, self.y_train)

    def predict(self, X_test, K):
        """See Base.predict"""
        pred = []
        for x0 in X_test:
            neighbours = [self._nearest_neighbours(self.root, x0, K)]
            pred.append(sum(n.value for n in neighbours) / K)
        return pred

    def _grow_tree(self, X_sub, y_sub, depth=0, parent=None):
        # Constants
        n = len(X_sub)
        split = (n - 1) // 2
        # Special case when no more points
        if not n:
            return None
        # Sort over target dimension
        dim = depth % self.P
        order = KDTree._argsort([x[dim] for x in X_sub])
        X_sort = [X_sub[i] for i in order]
        y_sort = [y_sub[i] for i in order]
        # Create node
        point = X_sort[split]
        value = y_sort[split]
        parent = Node(point, value, parent)
        # Recursion
        left = self._grow_tree(X_sort[:split], y_sort[:split],
                               depth + 1, parent)
        right = self._grow_tree(X_sort[(split + 1):], y_sort[(split + 1):],
                                depth + 1, parent)
        # Assign children
        parent.assign_children(left, right)

        return parent

    # Todo: expand for K > 1
    def _nearest_neighbours(self, node, x_test, K, depth=0):
        # Constants
        dim = depth % 2
        # Special case
        if node is None:
            return None
        # Recursion
        if x_test[dim] < node.point[dim]:
            target = node.left
            other = node.right
        else:
            target = node.right
            other = node.left
        target_candidate = self._nearest_neighbours(target, x_test,
                                                    K, depth + 1)
        # Compare to current node
        candidate, r2 = KDTree._closest(target_candidate, node, x_test)
        # Consider other branch
        d2 = (node.point[dim] - x_test[dim]) ** 2
        if r2 >= d2:
            other_candidate = self._nearest_neighbours(other, x_test,
                                                       K, depth + 1)
            candidate, __ = KDTree._closest(other_candidate, candidate, x_test)

        return candidate

    @staticmethod
    def _closest(n1, n2, x_test):
        # Special case
        if n1 is None:
            return n2, sum((x2 - x0)**2 for x0, x2 in zip(n2.point, x_test))
        if n2 is None:
            return n1, sum((x1 - x0)**2 for x0, x1 in zip(n1.point, x_test))
        # Compare distances
        d1 = sum((x1 - x0)**2 for x0, x1 in zip(n1.point, x_test))
        d2 = sum((x2 - x0)**2 for x0, x2 in zip(n2.point, x_test))
        return (n1, d1) if d1 < d2 else (n2, d2)

    @staticmethod
    def _argsort(seq):
        return sorted(range(len(seq)), key=seq.__getitem__)

class Node:

    def __init__(self, point, value, parent=None):
        self.point = point
        self.value = value
        self.parent = parent
        self.left = None
        self.right = None

    def assign_children(self, left, right):
        self.left = left
        self.right = right
