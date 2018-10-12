import numpy as np

def int_to_bin(x, dim=0):
    print("bin:",bin(x))
    x = bin(x)[2:]
    print("x:", x)
    k = dim - len(x)
    if k > 0:
        x = "0" * k + x
    return x

def rand(lower, upper):
    return (upper - lower) * np.random.random() + lower

if __name__ == '__main__':
    #a_int = int(np.random.random(0, 127))
    str = "0" * 3
    print(str)
    a_int = int(rand(0, 127))
    print(a_int)
    a = int_to_bin(a_int, dim=8)
    print(a)
    case = [[0, 1, 3, 4, 5, 6, 7, 8],
            [2, 3, 5, 4, 1, 3, 2, 1]]
    for i in range(8):
        x = np.array([[c[8 - i - 1] for c in case]])
        print(x)
    print("---------------------------------------------")
    a = np.array([int(t) for t in a])
    print(a)
    d = np.array([[1]])
    print(d.T)
    h = np.array([[1, 2, 3]])
    j = np.array([[1, 2, 3],
                  [2, 3, 4],
                  [4, 5, 7]])
    k = np.dot(h, j)
    print("------------------------------")
    print(k)