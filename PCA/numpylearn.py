import numpy as np

if __name__  == "__main__":
    arr = np.arange(4)
    print(arr)
    #np.random.shuffle(arr)
    #print(arr)

    X = np.array([[[5, 2], [4, 2]], [[1, 3], [2, 3]], [[1, 1], [0, 1]]])
    np.random.shuffle(X)
    #print(X)

    a = np.array([[2, 3, 4, 5, 6], [1, 3, 5, 4, 7], [3, 4, 5, 8, 9]])
    c = np.linalg.norm(a, ord=2, axis=-1)
    c = np.atleast_1d(c)

    #print(c==0)
    #print(c[c==0])
    #print(c)
    d = [[2],
         [3],
         [4]]
    #print(a / d)
    #np.random.shuffle(a)
    #print(a)
    #a = np.array([[1,2,3], [4,5,6]])
    #print(a.sum(axis=0))
    r = np.zeros((4, 3))
    print(r)
    amean = a.mean(axis=0)
    print(amean)
    print(a - amean)
    astd = a.std(axis=0)
    print(astd)

    y = np.array([[0, 0], [0, 0]])
    print(y.any())

    Y = np.empty([3, 3])
    print(Y)
    print("---------------------------------------------------------")
    K = [[1, 0, 0],
         [0, 2, 0],
         [0, 0, 3]]
    std_k = np.expand_dims(K, 1)
    #print(std_k)

    covariance_matrix = np.array([[1, 2, 0, 4],
                                  [3, 0, 5, 9],
                                  [6, 9, 8, 8],
                                  [4, 5, 1, 4]])
    i, j = np.linalg.eig(covariance_matrix)

    print(i.argsort()[::-1])
    idx = i.argsort()[::-1]
    print(i[idx][:2])
    print(j[:, idx][:, :2])
    print("---------------------------------")
    print(i)
    print(j)