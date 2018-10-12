import numpy as np

def rand(lower, upper):
    return (np.random.random() * (upper - lower)) + lower

def int_to_bin(x, dim=0):
    x_bin = bin(x)[2:]
    k = dim - len(x_bin)
    if k > 0:
        x_bin = k * "0" + x_bin
    return x_bin

def make_weight_matrix(m, n):
    weights = []
    for i in range(m):
        a = []
        for j in range(n):
            a.append(rand(-1, 1))
        a = np.array(a)
        weights.append(a)
    return np.array(weights)

def sigmoid(m):
    return 1 / (1 + np.exp(-m))

def sigmoid_derivative(y):
    return y * ( 1 - y )

def bin_to_int(x):
    out = 0
    for index, i in enumerate(reversed(x)):
        out += i * pow(2, index)
    return out

class Recurrent_Neural_Network():
    def __init__(self):
        self.input_num = 0
        self.output_num = 0
        self.hidden_num = 0

        self.input_weight = []
        self.output_weight = []
        self.hidden_weight = []

    def setup(self, input_num, output_num, hidden_num):
        self.input_num = input_num
        self.output_num = output_num
        self.hidden_num = hidden_num

        self.input_weight = make_weight_matrix(input_num, hidden_num)
        self.output_weight = make_weight_matrix(hidden_num, output_num)
        self.hidden_weight = make_weight_matrix(hidden_num, hidden_num)

    def predict(self, case, dim=0):
        guess = np.zeros(dim)
        hidden_history_layer = [np.zeros(self.hidden_num)]
        for i in range(dim):
            x = np.array([[c[dim - i - 1] for c in case]])

            hidden_layer = sigmoid(np.dot(x, self.input_weight) + np.dot(hidden_history_layer[-1], self.hidden_weight))
            output_layer = sigmoid(np.dot(hidden_layer, self.output_weight))
            hidden_history_layer.append(hidden_layer)
            guess[dim - i - 1] = np.round(output_layer[0][0])

        return guess

    def train(self, case, label, dim=0, learn=0.1):
        input_update = np.zeros_like(self.input_weight)
        output_update = np.zeros_like(self.output_weight)
        hidden_update = np.zeros_like(self.hidden_weight)

        guess = np.zeros_like(label)
        hidden_history_layer = [np.array([np.zeros(self.hidden_num)])]

        hidden_delta = 0
        error = 0
        output_deltas = []
        for i in range(dim):
            x = np.array([[c[dim - i -1] for c in case]])
            y = np.array([[label[dim - i - 1]]]).T
            hidden_layer = sigmoid(np.dot(x, self.input_weight) + np.dot(hidden_history_layer[-1], self.hidden_weight))
            output_layer = sigmoid(np.dot(hidden_layer, self.output_weight))

            output_error = y - output_layer
            output_deltas.append(output_error * sigmoid_derivative( output_layer ))
            hidden_history_layer.append(hidden_layer)

            #print("outputlayer1: ", np.round(output_layer[0][0]))
            guess[dim - i - 1] = np.round(output_layer[0][0])

        future_hidden_layer_delta = [np.zeros(self.hidden_num)]
        for i in range(dim):
            x = np.array([[c[i] for c in case]])
            hidden_layer = hidden_history_layer[-i-1]
            pev_hidden_layer = hidden_history_layer[-i-2]
            output_delta = output_deltas[-i-1]
            hidden_delta = (np.dot(output_delta, self.output_weight.T)
                            + np.dot(future_hidden_layer_delta, self.hidden_weight)) * sigmoid_derivative(hidden_layer)
            future_hidden_layer_delta = hidden_delta

            output_update += hidden_layer.T.dot(output_delta)
            hidden_update += pev_hidden_layer.T.dot(hidden_delta)
            input_update += x.T.dot(hidden_delta)

        self.input_weight += input_update * learn
        self.output_weight += output_update * learn
        self.hidden_weight += hidden_update * learn
        return guess

    def test(self):
        self.setup(2, 1, 16)
        for j in range(20000):
            a_int = int(rand(1, 127))
            a_bin = int_to_bin(a_int, dim=8)
            a = np.array([int(i) for i in a_bin])

            b_int = int(rand(1, 127))
            b_bin = int_to_bin(b_int, dim=8)
            b = np.array([int(i) for i in b_bin])

            c_int = a_int + b_int
            c_bin = int_to_bin(c_int, dim=8)
            c = np.array([int(i) for i in c_bin])

            guess = self.train([a, b], c, dim=8, learn=0.1)
            #if j % 1000 == 0 and j >= 1000:
            if j % 1000 == 0:
                #print(self.input_weight)
                #print(self.hidden_weight)
                #print(self.output_weight)
                print("a: ", a_bin)
                print("b: ", b_bin)
                print("c: ", c_bin)


                print("predict: ", guess)
                print("True: ", c_int)

                pre_result = bin_to_int(guess)
                print (str(a_int) + " + " + str(b_int) + " = " + str(pre_result))
                print("====================================================")

if __name__ == '__main__':
    nn = Recurrent_Neural_Network()
    nn.test()