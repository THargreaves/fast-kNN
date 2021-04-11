"""A naive approach to kNN, running in O(mndk) time."""

import math

from .base import Base


class KDTree(Base):

    def __init__(self, X_train, y_train):
        super().__init__(X_train, y_train)
        # Pre-computation of KD tree
        self.root = self._grow_tree(self.X_train, self.y_train)

    def predict(self, X_test, K):
        """See Base.predict"""
        pred = []
        for x0 in X_test:
            neighbours = self._nearest_neighbours(self.root, x0, K)
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

    def _nearest_neighbours(self, node, x_test, K, depth=0):
        # Constants
        dim = depth % 2
        # Special case
        if node is None:
            return [None for __ in range(K)]
        # Recursion
        if x_test[dim] < node.point[dim]:
            target = node.left
            other = node.right
        else:
            target = node.right
            other = node.left
        target_candidates = self._nearest_neighbours(target, x_test,
                                                     K, depth + 1)
        # Compare to current node
        candidates = KDTree._insert(target_candidates, node, x_test)
        # Consider other branch
        c = target_candidates[K-1]
        d2 = (node.point[dim] - x_test[dim]) ** 2
        if c is None or sum((x1 - x0)**2 for x0, x1 in zip(c.point, x_test)) >= d2:
            other_candidates = self._nearest_neighbours(other, x_test,
                                                        K, depth + 1)
            candidates = KDTree._merge(other_candidates, candidates, x_test, K)

        return candidates

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
    def _insert(candidates, n, x_test):
        if n is None:
            return candidates
        dn = sum((x1 - x0) ** 2 for x0, x1 in zip(n.point, x_test))
        replace = False
        for k in range(len(candidates)):
            if candidates[k] is None:
                replace = True
            else:
                c = candidates[k]
                d = sum((x1 - x0) ** 2 for x0, x1 in zip(c.point, x_test))
                if dn < d:
                    replace = True
            if replace:
                candidates.insert(k, n)
                del candidates[len(candidates) - 1]
                break
        return candidates

    @staticmethod
    def _merge(candidates1, candidates2, x_test, K):
        dists1 = [
            sum((x1 - x0) ** 2 for x0, x1 in zip(n.point, x_test)) if n else None
            for n in candidates1
        ]
        dists2 = [
            sum((x1 - x0) ** 2 for x0, x1 in zip(n.point, x_test)) if n else None
            for n in candidates2
        ]
        i, j = 0, 0
        candidates = []
        for k in range(K):
            if dists1[i] is None:
                candidates.extend(candidates2[j:j+(K-k)])
                break
            elif dists2[j] is None:
                candidates.extend(candidates1[i:i + (K - k)])
                break
            if dists1[i] < dists2[j]:
                candidates.append(candidates1[i])
                i += 1
            else:
                candidates.append(candidates2[j])
                j += 1
        return candidates

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
