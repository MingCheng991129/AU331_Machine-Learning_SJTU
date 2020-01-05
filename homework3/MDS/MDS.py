import pandas as pd
import numpy as np

import getMNISTData

# import math
import matplotlib.pyplot as plt



def getDij(data):
    Dij = np.zeros((data.shape[0], data.shape[0]), dtype = float)

    ## calculate step by step
    square = np.square(data)
    S2 = np.sum(square, axis = 1) 

    multiply = data.dot(data.T)
    multiply = -2 * multiply

    # print(multiply.shape)
    # print(S2.shape)
    add = np.add(multiply, S2)

    addition = multiply + S2
    # print(addition.shape)
    Dij = addition.T + S2
    # print(Dij.shape)
    return Dij

def MDS(data, axis):
    Dij = getDij(data)

    # sum1 = 0
    # sum2 = 0
    # Di = np.zeros((Dij.shape[0], 1))
    # Dj = np.zeros((Dij.shape[1], 1))
    
    # for i in range(Dij.shape[0]):
    #     sum1 = 0
    #     for j in range(Dij.shape[1]):
    #         sum1 += Dij[i, j]
    #     Di[i] = sum1

    # Dij = Dij.T 

    # for i in range(Dij.shape[0]):
    #     sum2 = 0
    #     for j in range(Dij.shape[1]):
    #         sum2 += Dij[i, j]
    #     Dj[i] = sum2

    # Dij = Dij.T 


    Di = np.sum(Dij, axis=1, keepdims = True)
    Dj = np.sum(Dij, axis=0, keepdims = True)


    ## calculate step by step
    item1 = Di / data.shape[0]
    item2 = Dj / data.shape[0]
    item3 = Dij
    sum_res = np.sum(Dij)

    D = np.ones((data.shape[0], data.shape[0]))
    D *= sum_res

    item4 = D / data.shape[0] ** 2
    # item5 = item4 ** 2

    ## calculate Bij
    Bij = 0.5 * (item1 + item2 - item3 - item4)
    # Bij = 0.5 * (Di / data.shape[0] + Dj / data.shape[0] - Dij - D / data.shape[0] ** 2)

    W , V = np.linalg.eig(Bij) ## complex number 
    # print(W)
    # print(V)

    idx = np.argsort(W).astype(np.int32) ## convert to int
    # print(type(idx))
    # idx.astype(np.int32)

    ## select new index
    new_idx = idx[-axis:]

    ## get new V and W
    V = V[:,new_idx]
    A = W[new_idx]
    
    # sqrt = np.sqrt(A)
    # res = V * sqrt

    result = V * A ** (0.5)

    return result


def plot(result):

    plt.figure(1)
    plt.scatter(result[:,0], result[:,1], c = label[:1000])
    plt.show()


if __name__ == '__main__':

    getMNISTData.img2csv("MNIST/train-images-idx3-ubyte", "MNIST/train-labels-idx1-ubyte",\
        "mnist.csv", num = 60000)
    dataset , label = getMNISTData.loadData(filename = 'mnist.csv') 

    Dij = getDij(dataset)
    result = MDS(dataset[:1000, :], axis = 2)
    plot(result)