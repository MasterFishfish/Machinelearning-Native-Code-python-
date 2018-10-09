import numpy as np
#随机打乱数据
def shuffle_data(X, Y):
    idx = np.arange(X.shape[0])
    np.random.shuffle(idx)
    return X[idx], Y[idx]

#划分训练集和测试集
def train_test_split(X, Y, test_size = 0.2, shuffle=True, seed=None):
    if shuffle:
        Xs, Ys = shuffle_data(X, Y)
    n_train_samples = int((X.shape[0]) * (1 - test_size))
    X_train, X_test = X[:n_train_samples], X[n_train_samples:]
    Y_train, Y_test = Y[:n_train_samples], Y[n_train_samples:]
    return X_train, X_test, Y_train, Y_test

#计算协方差矩阵
def caculate_covariance_matrix(X):
    n_samples = X.shape[0]
    covariance_matrix = (1 / n_samples) * ((X - X.mean(axis=0)).T.dot(X - X.mean(axis=0)))
    return np.array(covariance_matrix, dtype=float)

#正规化数据
def normalize(X, axis=-1, p=2):
    lp_norm = np.atleast_1d(np.linalg.norm(X, p, axis))
    lp_norm[lp_norm == 0] = 1
    return X / (np.expand_dims(lp_norm, axis))

#标准化数据
def standardize(X):
    X_std = np.zeros(X.shape)
    mean = X.mean(axis=0)
    for i in range(X.shape[1]):
        X_std[:, i] = X[:, i] - mean[i]
    return X_std

#PCA类
class PCA():
    def __init__(self, K):
        self.egien_values = None
        self.egien_vectors = None
        self.K = K

    def transform(X, self):
        covariance_matrix = caculate_covariance_matrix(X)

        self.egien_values, self.egien_vectors = np.linalg.eig(covariance_matrix)

        idx = self.egien_values.argsort()[::-1]
        egien_values = self.egien_values[idx][:self.K]
        egien_vectors = self.egien_vectors[:, idx][:, self.K]
        #egien_vectors这个是特征矩阵，np.linalg.eig计算出来的矩阵，每一列为一个特征值对应的特征向量

        transform_matrix = X.dot(egien_vectors)
        #返回特征值和降维之后的数据
        return egien_values, transform_matrix

def main():
    pass

if __name__ == "__main__":
    main()