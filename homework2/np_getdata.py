from sklearn.datasets import load_iris
import numpy as np

def getData():
    ### load iris data
    iris_data = load_iris()

    data = iris_data['data']
    labels = iris_data['target']

    ### get random data
    idx = list(range(data.shape[0]))
    np.random.shuffle(idx)

    data = data[idx]
    labels = labels[idx]

    ### normalize
    data /= np.max(abs(data))
    data -= np.mean(data, axis= 0)

    train_num = int(data.shape[0] * 0.8)
    test_num = data.shape[0] - train_num

    train_data = data[0:train_num]
    test_data = data[train_num:]

    train_label = labels[0:train_num]
    test_label = labels[train_num:]

    return train_data, test_data, train_label, test_label, data, labels, train_num
