import numpy as np
import nltk, itertools, csv
import operator
#nltk.download()

TXTCODING = 'utf-8'
unknown_token = 'UNKNOWN_TOKEN'
start_token = 'START_TOKEN'
end_token = 'END_TOKEN'

def sigmoid(X):
    z = np.array(X, dtype=float)
    print(z)
    return (1 / ( 1 + np.exp(-z)))

def sigmoid_derivative(y):
    return y * ( 1 - y )

def softmax(Z):
    exp_z = np.exp(Z)
    denominator = np.sum(exp_z, axis=-1)
    a = exp_z / denominator
    return a

def make_rand(lower, upper):
    return (np.random.random() * ( upper - lower )) + lower

def make_rand_weights_matirx(shape):
    row = shape[0]
    col = shape[1]
    weights = []
    for i in range(row):
        weight = []
        for j in range(col):
            weight.append(make_rand(-1, 1))
        weights.append(weight)
    return np.array(weights)

def tanh(X):
    z = np.array(X, dtype=float)
    return (np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z))

def tanh_derivative(y):
    return 1 - y * y

def LeakyReLU(z):
    b = 0
    if z[0, 0] > 0:
        b = float(z)
    else:
        b = float(0.01 * z)
    return b

def LeakyReLU_derivative(z):
    d = 0
    if z[0, 0] > 0:
       d = 1
    else:
        d = 0.01
    return d

#文本预处理的类，用于处理文本
class tokenFile2vector:
    def __init__(self, file_path, dict_size):
        self.file_path = file_path
        self.dict_size = dict_size

    # 将文本拆成句子，并加上句子开始和结束标志
    def get_sentences(self):
        sents = []
        with open(self.file_path, 'r', encoding='gbk', errors='ignore') as f:
            reader = csv.reader(f, skipinitialspace=True)
            a = 0
            for line in reader:
                sents_incow = nltk.sent_tokenize(line[0].lower())
                for sent_incow in sents_incow:
                    # sent_incow = sent_incow.split(" ")
                    # print(sent_incow)
                    # print(type(sent_incow))
                    if a != 0:
                        sent = nltk.word_tokenize(sent_incow)
                        sent.insert(0, start_token)
                        sent.append(end_token)
                        #print(sent)
                        sents.append(sent)
                a += 1
        return sents

    # 得到每句话的单词，并得到字典及字典中每个词的下标
    def get_dict_wordsIndex(self, sents):
        sent_words = sents
        word_freq = nltk.FreqDist(itertools.chain(*sent_words))
        print ('Get {} words.'.format(len(word_freq)))

        #most_common 返回出现次数最频繁的词及其频度(由于dict-size = 8000, 所以是前8000个)
        common_words = word_freq.most_common(self.dict_size-1)

        # 生成词典,词典中包含了前8000个词
        #由于之前的common_words是按照词的频度排列好的list, 所以dict_words是按词频降序排列的词的list
        #dict_words的某个元素的序列号 表示该词在数据集中第几常用，
        dict_words = [word[0] for word in common_words]
        dict_words.append(unknown_token)

        # 得到每个词的下标，用于生成词向量, 该字典可以通过每一个词检索到该词在整体中是第几常用的
        index_of_words = dict((word, ix) for ix, word in enumerate(dict_words))
        return sent_words, dict_words, index_of_words

    # 得到训练数据
    def get_vector(self):
        sents = self.get_sentences()
        sent_words, dict_words, index_of_words = self.get_dict_wordsIndex(sents)

        # 将每个句子中没包含进词典dict_words中的词替换为unknown_token
        for i, words in enumerate(sent_words):
            sent_words[i] = [w if w in dict_words else unknown_token for w in words]

        X_train = np.array([[index_of_words[w] for w in sent[:-1]] for sent in sent_words])

        print(X_train[23])
        # print(X_train[456])
        y_train = np.array([[index_of_words[w] for w in sent[1:]] for sent in sent_words])
        print(y_train[23])
        return X_train, y_train, dict_words, index_of_words

#类LSTM，用于训练dataset
class LSTM():
    def __init__(self):
        self.input_num = 0
        self.hidden_num = 0
        self.output_num = 0

        self.fg_hidden_weights = np.array([])
        self.ig_hidden_weights = np.array([])
        self.og_hidden_weights = np.array([])
        self.ag_hidden_weights = np.array([])

        self.fg_input_weights = np.array([])
        self.ig_input_weights = np.array([])
        self.og_input_weights = np.array([])
        self.ag_input_weights = np.array([])

        self.output_weights = np.array([])

        self.fg_bias = 0
        self.ig_bias = 0
        self.og_bias = 0
        self.ag_bias = 0

    def setup(self, input_num, hidden_num, output_num):
        self.input_num = input_num
        self.hidden_num = hidden_num
        self.output_num = output_num

        hidden_shape = [hidden_num, hidden_num]
        input_shape = [input_num, hidden_num]
        self.fg_hidden_weights = make_rand_weights_matirx(hidden_shape)
        self.ig_hidden_weights = make_rand_weights_matirx(hidden_shape)
        self.og_hidden_weights = make_rand_weights_matirx(hidden_shape)
        self.ag_hidden_weights = make_rand_weights_matirx(hidden_shape)

        self.fg_input_weights = make_rand_weights_matirx(input_shape)
        self.ig_input_weights = make_rand_weights_matirx(input_shape)
        self.og_input_weights = make_rand_weights_matirx(input_shape)
        self.ag_input_weights = make_rand_weights_matirx(input_shape)

        self.fg_bias = make_rand(-1, 1)
        self.ig_bias = make_rand(-1, 1)
        self.og_bias = make_rand(-1, 1)
        self.ag_bias = make_rand(-1, 1)

        self.output_weights = make_rand_weights_matirx([self.hidden_num, self.output_num])

    def predict(self, X, dict_words):
        dim = len(X)
        prior_state = []
        init_state = np.array([[0] * self.hidden_num])
        prior_state.append(init_state)
        prior_hidden = []
        init_hidden = np.array([[0] * self.hidden_num])
        prior_hidden.append(init_hidden)
        outputs = []
        words = []
        for i in range(dim):
            prev_hidden = prior_hidden[-1]
            prev_state = prior_state[-1]
            x = np.array([[X[i]]])
            forget_matrix = sigmoid(
                np.dot(x, self.fg_input_weights) + np.dot(prev_hidden, self.fg_hidden_weights) + self.fg_bias)
            input_matrix = sigmoid(
                np.dot(x, self.ig_input_weights) + np.dot(prev_hidden, self.ig_hidden_weights) + self.ig_bias)
            a_matrix = sigmoid(
                np.dot(x, self.ag_input_weights) + np.dot(prev_hidden, self.ag_hidden_weights) + self.ag_bias)
            output_matrix = sigmoid(
                np.dot(x, self.og_input_weights) + np.dot(prev_hidden, self.og_hidden_weights) + self.og_bias)

            #更新hidden的状态
            this_tanh = tanh(
                np.multiply(prev_state, forget_matrix) + np.multiply(a_matrix, input_matrix))
            hidden = this_tanh * output_matrix

            prior_hidden.append(hidden)

            # 更新state的状态
            next_state = np.multiply(prev_state, forget_matrix) + np.multiply(a_matrix, input_matrix)
            prior_state.append(next_state)

            # 输出本次的结果
            output_z = np.dot(hidden, self.output_weights)
            output = LeakyReLU(output_z)
            word = dict_words[int(np.round(output[0, 0]))]
            outputs.append(output)
            words.append(word)
        return outputs, words

    def train(self, X, Y, learn_rate=0.1):
        #先正向传播
        dim = len(Y)
        prior_state = []
        init_state = np.array([[0] * self.hidden_num])
        prior_state.append(init_state)
        prior_hidden = []
        init_hidden = np.array([[0] * self.hidden_num])
        prior_hidden.append(init_hidden)
        outputs = []

        tanhs = []
        output_gates = []
        input_gates = []
        forget_gates = []
        a_gates = []

        errors = []
        output_deltas = []

        for i in range(dim):
            state = prior_state[-1]
            prev_hidden = prior_hidden[-1]
            x = np.array([[X[i]]])
            y = np.array([[Y[i]]])
            forget_matrix = sigmoid(
                np.dot(x, self.fg_input_weights) + np.dot(prev_hidden, self.fg_hidden_weights) + self.fg_bias)
            input_matrix = sigmoid(
                np.dot(x, self.ig_input_weights) + np.dot(prev_hidden, self.ig_hidden_weights) + self.ig_bias)
            a_matrix = sigmoid(
                np.dot(x, self.ag_input_weights) + np.dot(prev_hidden, self.ag_hidden_weights) + self.ag_bias)
            output_matrix = sigmoid(
                np.dot(x, self.og_input_weights) + np.dot(prev_hidden, self.og_hidden_weights) + self.og_bias)

            # 更新hidden, 以及各门的状态
            this_tanh = tanh(
                np.multiply(state, forget_matrix) + np.multiply(a_matrix, input_matrix))
            hidden = this_tanh * output_matrix

            prior_hidden.append(hidden)
            tanhs.append(this_tanh)
            output_gates.append(output_matrix)
            forget_gates.append(forget_matrix)
            a_gates.append(a_matrix)
            input_gates.append(input_matrix)

            # 更新state的状态
            next_state = np.multiply(state, forget_matrix) + np.multiply(a_matrix, input_matrix)
            prior_state.append(next_state)

            # 输出本次的结果
            output_z = np.dot(hidden, self.output_weights)
            output = LeakyReLU(output_z)
            outputs.append(output)

            #添加本次的error
            error = y - output
            errors.append(error)
            output_deltas.append(error * LeakyReLU_derivative(output_z))


        output_updates = np.zeros_like(self.output_weights)
        ig_input_updates = np.zeros_like(self.ig_input_weights)
        fg_input_updates = np.zeros_like(self.fg_input_weights)
        ag_input_updates = np.zeros_like(self.ag_input_weights)
        og_input_updates = np.zeros_like(self.og_input_weights)

        ig_hidden_updates = np.zeros_like(self.ig_hidden_weights)
        fg_hidden_updates = np.zeros_like(self.fg_hidden_weights)
        ag_hidden_updates = np.zeros_like(self.ag_hidden_weights)
        og_hidden_updates = np.zeros_like(self.og_hidden_weights)

        fb_updates = 0
        ib_updates = 0
        ob_updates = 0
        ab_updates = 0
        #反向传播SGD更新权值
        for j in range(dim):
            output = outputs[-j-1]
            output_gate = output_gates[-j-1]
            forget_gate = forget_gates[-j-1]
            input_gate = input_gates[-j-1]
            a_gate = a_gates[-j-1]
            this_tanh = tanhs[-j-1]
            output_delta = output_deltas[-j-1]
            this_hidden = prior_hidden[-j-1]
            prev_hidden = prior_hidden[-j-2]
            this_state = prior_state[-j-1]
            prev_state = prior_state[-j-2]
            x = np.array([[X[-j-1]]])
            y = np.array([[Y[-j-1]]])

            hidden_delta = np.dot(output_delta, self.output_weights.T)
            output_gate_delta = (hidden_delta * this_tanh) * sigmoid_derivative(output_gate)
            state_delta = (hidden_delta * output_gate) * tanh_derivative(this_tanh)
            forget_gate_delta = (state_delta * prev_state) * sigmoid_derivative(forget_gate)
            a_gate_delta = (state_delta * input_gate) * tanh_derivative(a_gate)
            input_gate_delta = (state_delta * a_gate) * sigmoid_derivative(input_gate)

            output_updates += np.dot(this_hidden.T, output_delta)
            fg_input_updates += np.dot(x.T, forget_gate_delta)
            fg_hidden_updates += np.dot(prev_hidden.T, forget_gate_delta)
            ig_input_updates += np.dot(x.T, input_gate_delta)
            ig_hidden_updates += np.dot(prev_hidden.T, input_gate_delta)
            og_input_updates += np.dot(x.T, output_gate_delta)
            og_hidden_updates += np.dot(prev_hidden.T, output_gate_delta)
            ag_input_updates += np.dot(x.T, a_gate_delta)
            ag_hidden_updates += np.dot(prev_hidden.T, a_gate_delta)

            fb_updates += forget_gate_delta
            ib_updates += input_gate_delta
            ob_updates += output_gate_delta
            ab_updates += a_gate_delta

        self.fg_input_weights = self.fg_input_weights + learn_rate * fg_input_updates
        self.fg_hidden_weights = self.fg_hidden_weights + learn_rate * fg_hidden_updates
        self.ig_input_weights = self.ig_input_weights + learn_rate * ig_input_updates
        self.ig_hidden_weights = self.ig_hidden_weights + learn_rate * ig_hidden_updates
        self.og_input_weights = self.og_input_weights + learn_rate * og_input_updates
        self.og_hidden_weights = self.og_hidden_weights + learn_rate * og_hidden_updates
        self.ag_input_weights = self.ag_input_weights + learn_rate * ag_input_updates
        self.ag_hidden_weights = self.ag_hidden_weights + learn_rate * ag_hidden_updates

        self.output_weights = self.output_weights + learn_rate * output_updates

        self.fg_bias += learn_rate * fb_updates
        self.ig_bias += learn_rate * ib_updates
        self.og_bias += learn_rate * ob_updates
        self.ag_bias += learn_rate * ab_updates

    def do_train(self, X, Y):
        train_num = int(len(X) * 0.8)
        for j in range(1000):
            for i in range(train_num):
                self.train(X[i], Y[i])

    def test(self, X, Y, dict_words):
        test_num = int(len(X) * 0.2)
        train_num = int(len(X) * 0.8)
        error_num = 0
        for i in range(test_num):
            outputs, words = self.predict(X[train_num + i], dict_words)
            if operator.eq(outputs, Y[train_num + i]):
                error_num += 1
            if i % 500 == 0 and i >= 500:
                print(words)
        print(error_num / test_num)


if __name__ == "__main__":
    lstm = LSTM()
    lstm.setup(1, 100, 1)

    dict_size = 8000
    file_path = r"./dataset.csv"
    myTokenFile = tokenFile2vector(file_path, dict_size)
    X, Y, dict_words, index_of_words = myTokenFile.get_vector()

    lstm.do_train(X, Y)
    lstm.test(X, Y, dict_words)