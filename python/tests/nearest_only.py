import os
import random
import sys
import time

from sklearn.neighbors import KNeighborsRegressor

sys.path.append(os.path.abspath(os.path.join('..', 'implementations')))
from implementations import implementations


if __name__ == "__main__":
    N = 500  # training set size
    M = 100  # test set size
    D = 2  # dimension of data
    K = 1  # number of neighbours to use
    SEED = 42  # random seed
    TOL = 10e-16

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

    mod = KNeighborsRegressor(K).fit(X_train, y_train)
    truth = mod.predict(X_test)

    for imp in implementations:
        Imp = imp(X_train, y_train)
        print(f"Predicting with {Imp.__str__()}...", end='')
        pred = Imp.predict(X_test, K)
        correct = all(abs(p - t) < TOL for p, t in zip(pred, truth))
        print("pass" if correct else "fail")
