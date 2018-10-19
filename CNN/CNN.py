import numpy as np

def sigmoid(x):
    return 1 / ( 1 + np.exp(-x))

def sigmoid_derivative(y):
    return y * ( 1 - y )

def softmax(z):
    exp_z = np.exp(z)
    #softmax分母
    softmax_denominator = np.sum(exp_z, axis=-1)
    #softmax分子
    #softmax的值
    a = exp_z / softmax_denominator
    return a

def ouput_delta(a, y,  idx):
    pass

def rand(lower, upper):
    return (np.random.random() * (upper - lower)) + lower

#m决定了矩阵有多少行， n决定了矩阵有多少列
def make_rand_matrix(m, n):
    weights = []
    for i in range(m):
        a = []
        for j in range(n):
            a.append(rand(-1, 1))
        weights.append(a)
    weights = np.array(weights)
    return weights

def load_dataset():
    pass

#case必须是numpy.narray
def add_padding(padding, case):
    #添加最前面和后面的列0
    x = case
    for i in range(padding):
        c1 = np.array([[0] * (x.shape[0])])
        x = np.hstack((c1.T, x))
        x = np.hstack((x, c1.T))
    #添加顶部或者底部的行0
    for j in range(padding):
        c2 = np.array([[0] * (x.shape[1])])
        x = np.row_stack((c2, x))
        x = np.row_stack((x, c2))
    return x

#该函数用于求矩阵所有元素的和
def matrix_allsum(x):
    y = np.reshape(x, (x.size, ))
    return np.sum(y, axis=-1)

#该函数用于求矩阵的卷积, 此处步长为1
def matrix_convolution(case, kernel_size, conv_kernel):
    row = case.shape[0]
    column = case.shape[1]
    conv_results = []
    for i in range(row - kernel_size[0] + 1):
        conv_result = []
        for j in range(column - kernel_size[1] + 1):
            part_matrix = case[i:i + kernel_size[0], j:j + kernel_size[1]]
            part_data = matrix_allsum(np.multiply(part_matrix, conv_kernel))
            conv_result.append(part_data)
        conv_results.append(conv_result)
    return np.array(conv_results)

#该函数用于矩阵的池化，原矩阵无重复的池化部分
def matrix_Averagepooling(case, pooling_size):
    average_num = float(1 / np.zeros(pooling_size).size)
    row = case.shape[0]
    column = case.shape[1]
    pooling_results = []
    i = 0
    while i <= (row - pooling_size[0]):
        i += pooling_size[0]
        pooling_result = []
        j = 0
        while j <= (column - pooling_size[1]):
            part_matrix = case[i:i + pooling_size[0], j:j + pooling_size[1]]
            part_data = matrix_allsum(np.multiply(part_matrix, average_num))
            pooling_result.append(part_data)
            j += pooling_size[1]
        pooling_results.append(pooling_result)
    return np.array(pooling_results)

#该函数将下采样池化后的矩阵还原成为和原矩阵形状相同的上采样矩阵
def pooling_origin(case, pooling_size):
    x = case
    row = x.shape[0]
    column = x.shape[1]
    m = pooling_size.shape[0]
    n = pooling_size.shape[1]
    origin_matrix = []

    i = 0
    a = 0
    while i < row:
        vector =[]
        j = 0
        b = 0
        while j < column:
            data = float(x[i][j] / (m * n))
            vector.append(data)
            b += 1
            if (b % n) == 0:
                j += 1
        origin_matrix.append(vector)
        a += 1
        if (a % m) == 0:
            i += 1

    return np.array(origin_matrix)

#用现成的数据集进行手写数字的识别
#只有1个卷积层和1个池化层的，1个全连接层的CNN
#池化层使用的是下采样算法
class CNN():
    def __init__(self):
        #Convolution kernel size
        self.convkernel_num = 0
        self.convkernel_size = []
        self.conv_padding = 0
        #暂时只写步长为1的情况
        self.conv_sride = 0
        #pooling kernel size
        self.poolingkernel_size = []

        self.hidden_num = 0
        self.input_num = 0
        self.output_num = 0

        self.conv_kernel = np.array([0])
        #由于采用的是下采样算法池化，所以不用使用矩阵来表示pooling_kernel
        self.pooling_kernel = np.array([0])
        self.hidden_weight = np.array([0])
        self.input_weight = np.array([0])
        self.output_weight = np.array([0])

    def setup(self, X, convkernel_size, convkernel_num, conv_padding, poolingkernel_size, conv_stride, hidden_num):
        self.convkernel_size = convkernel_size
        self.convkernel_num = convkernel_num
        self.conv_padding = conv_padding
        self.poolingkernel_size = poolingkernel_size

        self.input_num = ((((X.shape[0] + 2 * conv_padding - convkernel_size) / conv_stride) + 1) / poolingkernel_size) ** 2
        self.output_num = 10
        self.hidden_num = hidden_num

        self.conv_kernel = make_rand_matrix(convkernel_size[0], convkernel_size[1])
        self.input_weight = make_rand_matrix(self.input_num, hidden_num)
        self.hidden_weight = make_rand_matrix(self.hidden_num, self.output_num)

    def identifty(self, X):
        #conv
        kernel_size = self.convkernel_size
        case = add_padding(self.conv_padding, X)
        conv_results = matrix_convolution(case, kernel_size, self.conv_kernel)

        #pooling(Average_pooling)
        pooling_size = self.poolingkernel_size
        pooling_reults = matrix_Averagepooling(conv_results, pooling_size)

        #FC
        input_data = np.reshape(pooling_reults, (1, pooling_reults.size))
        hidden_layer = sigmoid(np.dot(input_data, self.input_weight))
        output_layer = sigmoid(np.dot(hidden_layer, self.hidden_weight))
        a = softmax(output_layer)

        return a

    def train(self, X, Y, learn=0.1):
        input_update = np.zeros_like(self.input_weight)
        hidden_update = np.zeros_like(self.hidden_weight)
        conv_update = np.zeros_like(self.conv_kernel)
        #正向传播
        #conv
        kernel_size = self.convkernel_size
        #加上padding
        case = add_padding(self.conv_padding, X)
        conv_results = matrix_convolution(case, kernel_size, self.conv_kernel)
        # pooling(Average_pooling)
        pooling_size = self.poolingkernel_size
        pooling_reults = matrix_Averagepooling(conv_results, pooling_size)
        # FC
        input_data = np.reshape(pooling_reults, (1, pooling_reults.size))
        hidden_layer = sigmoid(np.dot(input_data, self.input_weight))
        output_layer = sigmoid(np.dot(hidden_layer, self.hidden_weight))
        a = softmax(output_layer)

        #反向传播
        #反向传播到hidden层
        output_delta = Y - a
        hidden_delta = (np.dot(output_delta, self.hidden_weight.T)) * sigmoid_derivative(hidden_layer)
        input_delta = np.dot(hidden_delta, self.input_weight.T)
        #反向传播到pooling层
        input_delta = np.reshape(input_delta, (pooling_reults.shape[0], pooling_reults.shape[1]))
        pooling_delta = pooling_origin(input_delta, pooling_size)

        #梯度下降
        hidden_update += np.dot(hidden_layer.T, output_delta)
        input_update += np.dot(hidden_delta.T, input_data)
        conv_update += matrix_convolution(case, pooling_delta.shape, pooling_delta)

        self.input_weight += input_update * learn
        self.hidden_weight += hidden_update * learn
        self.conv_kernel += conv_update * learn

        return a

    def do_train(self):
        #录入数据集
        #进行随机梯度下降训练
        pass
    def test(self):
        #录入测试集
        #进行错误率的检测
        pass

if __name__ == '__main__':
    pass

