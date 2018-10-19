import numpy as np


if __name__ == '__main__':
    a = np.array([[0, 1, 2]])
    b = np.array([[1, 2, 3, 4, 5]])
    print(np.exp(b))
    print(np.sum(np.exp(b), axis=-1))
    print(np.exp(b) / np.sum(np.exp(b),axis=-1))
    print(b.shape)
    c = np.array([[1, 3, 4, 5, 6],
                  [2, 3, 1, 4, 5],
                  [12,2, 5, 6, 15],
                  [9, 6, 7, 4, 6],
                  [19,2, 3, 3, 1]])
    print(np.multiply(c, 2))
    print(np.sum(b, axis=-1))
    c1 = np.array([[0] * 5])
    c1 = c1.T
    c2 = np.hstack((c1, c))
    c3 = np.hstack((c, c1))
    print(c2)
    print("=====================")
    print(c3)
    c4 = np.hstack((c2, c1))
    print("=====================")
    print(np.sum(np.reshape(c4, (1, c4.size)), axis=-1))
    print("========================")
    d = np.array([[1, 1, 2, 2, 5, 5],
                  [1, 1, 2, 2, 5, 5],
                  [3, 3, 4, 4, 6, 6],
                  [3, 3, 4, 4, 6, 6],
                  [8, 8, 9, 9, 7, 7],
                  [8, 8, 9, 9, 7, 7]])
    #e = np.array([[1, 1, 2, 2, 1, 1, 2, 2, 3, 3, 4, 4, 3, 3, 4, 4]])
    e = np.reshape(d, (1, d.size ))
    print(d[0][0])
    print(np.reshape(e, (6, 6)))
    g = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9]])
    print(np.reshape(g, (3, 3)))

