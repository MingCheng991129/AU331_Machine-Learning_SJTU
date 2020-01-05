import pandas as pd 
import numpy as np 
import random

def getWMData():
    wm_data = pd.read_csv('watermelon3.0.csv', encoding = 'gbk', engine = 'python')
    wm_data = wm_data.drop(['密度', '含糖率'], axis = 1)
    # print(wm_data)
    ## get numpy data
    wm_data_np = np.array(wm_data)

    train_num = int(wm_data.shape[0] * 0.7)
    test_num = wm_data.shape[0] - train_num

    test_data = wm_data_np
    lis = [i for i in range(wm_data.shape[0])]

    # print(train_data)

    train_idx = random.sample(lis, train_num)
    test_data = np.delete(test_data, train_idx, axis = 0)

    test_idx = []
    train_data = wm_data_np
    for i in range(wm_data.shape[0]):
        if i not in train_idx:
            test_idx.append(i)
    train_data = np.delete(train_data, test_idx, axis = 0)
    # print(train_data)

    return train_data, test_data
    

