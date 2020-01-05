import pandas as pd 
import numpy as np 
import getWMData
import distance_function


def cluster(distance, feature, num):
    C = []

    for i in range(feature.shape[0]):
        # Ci = i.tolist()
        # Ci = list(i)
        Ci = [i]
        C.append(Ci)
    # print(C)

    # M = np.ones((feature.shape[0], feature.shape[0]))
    M = np.zeros((feature.shape[0], feature.shape[0]))

    for i in range(feature.shape[0]):
        for j in range(i + 1, feature.shape[0]):
            if distance == 'Euclidean_distance':
                M[i, j] = distance_function.Euclidean_distance(C[i], C[j], feature)
                # print(M[i, j])

            elif distance == 'NCut_distance':
                M[i, j] = distance_function.NCut_distance(C[i], C[j], feature)
                # print(M[i, j])

            M[j, i] = M[i, j]
    # M = M.T 

    if distance == 'Euclidean_distance':
        print('Euclidean_distance:')
    elif distance == 'NCut_distance':
        print('NCut_distance:')

    row = feature.shape[0]
    while row > num:
    # while row >= num:
    
        print(row, ' cluster:')
        print(C)
        
        new_i, new_j = distance_function.findShortest(C, M)
        # print(new_i, new_j)
        C[new_i].extend(C[new_j])
        # print(C)
        # print(C[new_i])
        ## 
        ## don't forget!!!
        C.remove(C[new_j])
        M = np.delete(M, new_j, axis = 0)
        M = np.delete(M, new_j, axis = 1)

        # M = M.delete(new_j, axis = 0)
        # M = M.delete(new_j, axis = 1)
        # C = C.remove(C[new_j])

        for i in range(row - 1):
            if distance == 'Euclidean_distance':
                M[new_i, i] = distance_function.Euclidean_distance(C[new_i], C[i], feature)
                # print(M)
                # print(M[new_i, i])
            elif distance == 'NCut_distance':
                M[new_i, i] = distance_function.NCut_distance(C[new_i], C[i], feature)
                # print(M)
                # print(M[new_i, i])
            M[i, new_i] = M[new_i, i]
            # print(M)

        # for i in range(row - 1):
        #     if distance == 'Euclidean_distance':
        #         M[new_i, j] = Euclidean_distance(C[new_i], C[i], feature)

        #     elif distance == 'NCut_distance':
        #         M[new_i, j] = NCut_distance(C[new_i], C[i], feature)

        #     # print(M)

        row -= 1

    ## 2 num
    print('2 cluster:')
    print(C)
   


if __name__ == "__main__":

    feature, _ = getWMData.getWMData()

    cluster('Euclidean_distance', feature, 2)
    print('\n')
    cluster('NCut_distance', feature, 2)
                