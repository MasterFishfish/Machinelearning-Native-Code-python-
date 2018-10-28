from itertools import chain
from pythonML.pythonLearning.LSTM import LSTM
import numpy as np
if __name__ == "__main__":

    # dict_size = 8000
    # file_path = r"./dataset.csv"
    # myTokenFile = LSTM.tokenFile2vector(file_path, dict_size)
    # #X_train, y_train, dict_words, index_of_words = myTokenFile.get_vector()
    # sents = myTokenFile.get_sentences()
    # myTokenFile.get_vector()
    a = np.array([[1.8] * 8])
    lista = [[1, 2, 3, 4],
             [3, 4, 5, 6]]
    print(np.array(a, dtype=int))
    print(np.array([[a[0, 0]]]) > 0)
    if np.array([[a[0, 0]]]) <= 0:
        print("yyy")
    print(1 / (1 + np.exp(-432)))

