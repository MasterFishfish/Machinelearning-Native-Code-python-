
import matplotlib.pyplot as plt
import numpy as np
from numpy import *
from numpy.linalg import *

def PLA():
    W = np.ones(4)
    count = 0
    W[0] = -10
    dataset = np.array([[1, 0.10723, 0.64385, 0.29556, 1],
                     [1, 0.2418, 0.83075, 0.42741, 1],
                     [1, 0.23321, 0.81004, 0.98691, 1],
                     [1, 0.36163, 0.14351, 0.3153, -1],
                     [1, 0.46984, 0.32142, 0.000042772, -1],
                     [1, 0.25969, 0.87208, 0.075063, -1]])
    while True:
        count += 1
        for i in range(0,len(dataset)):
            X = dataset[i][:-1]
            Y = np.dot(W,X)
            if sign(Y) == sign(dataset[i][-1]):
                continue
            else:
                iscomplete = False
                W = W + (dataset[i][-1])*np.array(X)

        if iscomplete:
            break

    if iscomplete:
        print("the time is:",count)
        print("the W is:",W)

    return W

def main():
    PLA()

if __name__ == '__main__':
    main()
