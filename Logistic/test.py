import numpy as np

if __name__ == "__main__":
    feature = np.array([[1, 2, 3, 4],
                        [1, 3, 4, 7]])
    weights = np.array([[2],
                        [3],
                        [4],
                        [5]])

    print(np.dot(feature, weights))