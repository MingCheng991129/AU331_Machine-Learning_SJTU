import pandas as pd 
import numpy as np

def Euclidean_distance(center1, center2, feature):
    distance = 0

    ## traverse 
    for i in center1:
        for j in center2:

            ## calculate step by step
            square = np.square(feature[i]-feature[j])
            sum_res = np.sum(square)
            ## calculate the distance
            distance += np.sqrt(sum_res)

    length = len(center1) * len(center2)
    distance /= length

    return distance



def NCut_distance(center1, center2, feature):
    distance = 0
    W = np.zeros((feature.shape[0], feature.shape[0]))

    C = []
    # C.append(center1)
    # C.append(center2)
    C.extend(center1)
    C.extend(center2)
    # print(C)

    sigma = float(1)
    for i in C:
        for j in C:

            ## calculate step by step
            square = np.square(feature[i]-feature[j])
            sum_res = - np.sum(square)
            res = sum_res / (2 * sigma ** 2)

            ## get W matrix

            W[i, j] = np.exp(res)
    # print(W)


    # temp1 = np.ones((feature.shape[0], 1))
    # temp2 = np.ones((feature.shape[0], 1))

    temp1 = np.zeros((feature.shape[0], 1))
    temp2 = np.zeros((feature.shape[0], 1))

    ## set the value of 1 
    for i in center1:
        temp1[i] = 1
    for i in center2:
        temp2[i] = 1

    D = np.diag(np.sum(W, 1))
    # print(D)
    
    # for i in C:
    #     for j in C:
    #         W[i][j] = np.exp(-(np.sum(np.square(feature[i]-feature[j]))) \
    #               /(2*sigma**2))
    
    ## multiply step by step
    multiply1 = np.dot(temp1.T, W)
    cut = multiply1.dot(temp2)

    multiply2 = np.dot(temp1.T, D)
    vol1 = multiply2.dot(temp1)

    multiply3 = np.dot(temp2.T, D)
    vol2 = multiply3.dot(temp2)

    # print(multiply1)
    # print(multiply2)
    # print(multiply3)

    ## get cut and vol successfully

    ## set value of 1
    if len(center1) == 1:
        vol1 = 1
    if len(center2) == 1:
        vol2 = 1

    ## get the distance
    distance = cut / vol1 + cut / vol2

    return distance

def findShortest(C, M):
    min_ = 1e5 ## 

    new_i = -1
    new_j = -1
    m = len(C)

    # print(C)
    # print(len(C))

    for i in range(m):
        for j in range(m):

            ## shortest
            if M[i, j] < min_:
                if i != j:
                    min_ = M[i, j]

                    ## update new_i, new_j
                    new_i = i
                    new_j = j
                    # print(new_i)
                    # print(new_j)
                # else: 

    return new_i, new_j