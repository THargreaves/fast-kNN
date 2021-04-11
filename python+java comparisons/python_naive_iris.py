import pandas as pd
from os.path import dirname, abspath, join
import random
import sys
import time

sys.path.append(".")

from implementations import implementations

if __name__ == "__main__":

    iris = pd.read_csv("iris.csv")
    iris = iris.to_numpy()

    test_inds = [61,36,139,144,95,78,77,15,14,105,133,33,110,101,126,74,91,62,16, \
                 20,128,25,106,55,31,112,53,149,114,92,143,104,141,24,138,97,7, \
                 79,39,122,64,123,47,116,80,17,34,32,98,136] # fixed but random

    train_inds = [i for i in range(150) if i not in test_inds]

    Xtt = iris[test_inds, 0:4]
    Ytt = iris[test_inds, 4]

    Xtr = iris[test_inds, 0:4]
    Ytr = iris[test_inds, 4]

    naive = implementations[0]

    kNN = naive(Xtr, Ytr)
    preds = kNN.predict(Xtt, K=3)

    print(preds)
