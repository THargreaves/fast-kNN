"""A naive approach to kNN, running in O(mndk) time."""

# Imports
import random
import time

# Parameters
N = 5000  # training set size
M = 200   # test set size
D = 20  # dimension of data
K = 5  # number of neighbours to use
SEED = 42  # random seed

# Helpers
def distance(X1, X2):
    """Euclidean distance between two vectors."""
    return sum(x1 * x2 for x1, x2 in zip(X1, X2))

# Generate dataset
random.seed(SEED)
X_train = [[random.gauss(0, 1) 
            for d in range(D)]
           for n in range(N)]
y_train = [random.gauss(0, 1) for n in range(N)]
X_test = [[random.gauss(0, 1) 
           for d in range(D)]
          for n in range(M)]
y_pred = []

# Record start time
start = time.time()
  
# kNN implementation
selected = [False for i in range(N)]
for X0 in X_test:
    neighbour_values = []
    for k in range(K):
        neighest = None  # (i, X, y, d)
        for i, (X, y) in enumerate(zip(X_train[1:], y_train[1:])):
            if selected[i]:
                continue
            d = distance(X0, X)
            if neighest is None or d < neighest[2]:
                neighest = i, X, y, d
        selected[i] = True
        neighbour_values.append(y)
    y_pred.append(sum(neighbour_values) / K)

# Report elapsed time
elapsed = time.time() - start
print(f"Completed in {elapsed:.02f} seconds")

