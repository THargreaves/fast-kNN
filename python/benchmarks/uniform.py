import os
import random
import sys
import time

sys.path.append(os.path.abspath(os.path.join('..', 'implementations')))
from implementations import implementations


if __name__ == "__main__":
    N = 2500  # training set size
    M = 100  # test set size
    D = 2  # dimension of data
    K = 1  # number of neighbours to use
    SEED = 42  # random seed

    # Generate dataset
    random.seed(SEED)
    X_train = [[random.gauss(0, 1)
                for d in range(D)]
               for n in range(N)]
    y_train = [random.uniform(0, 1) for n in range(N)]
    X_test = [[random.uniform(0, 1)
               for d in range(D)]
              for n in range(M)]
    y_pred = []

    times = []
    for imp in implementations:
        Imp = imp(X_train, y_train)
        print(f"Predicting with {Imp.__str__()}")
        start = time.time()
        Imp.predict(X_test, K)
        end = time.time()
        times.append(end - start)

    print(times)
