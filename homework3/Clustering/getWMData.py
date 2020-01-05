import pandas as pd 
import numpy as np 


def getIndex(set_, target):
    lis = list(set_)
    for i in range(len(lis)):
        if target == lis[i]:
            break
    return i

def getWMData():
    wm_data = pd.read_csv('watermelon3.0.csv', encoding = 'gbk', engine = 'python')
    # print(wm_data)
    ## get numpy data
    wm_data = wm_data.drop(['好瓜','编号'], axis = 1) ## 17 * 9
    wm_data_np = np.array(wm_data)

    # print(wm_data_np.shape)
    # print(wm_data_np)

    ## one hot
    new_wm_data = np.zeros((wm_data_np.shape[0], 5 * 3 + 2 + 2), dtype = float)
    new_wm_data =  new_wm_data.T
    wm_data_np = wm_data_np.T ## 9 * 17
    # print(wm_data_np)

    unique_feat_lis = []
    for i in range(6):
        unique_feat = set(wm_data_np[i])
        unique_feat_lis.append(unique_feat)
    # print(unique_feat_lis)

    ans = 2
    # print(wm_data_np[0:6, :])
    for i in range(5):
        feature = wm_data_np[i]
        for j in range(17):
            # print(getIndex(unique_feat_lis[i], feature[j]), j, 'aaaaaa')
            if getIndex(unique_feat_lis[i], feature[j]) == 0:
                new_wm_data[ans, j] = 1

            elif getIndex(unique_feat_lis[i], feature[j]) == 1:
                new_wm_data[ans - 1, j] = 1

            elif getIndex(unique_feat_lis[i], feature[j]) == 2:
                new_wm_data[ans - 2, j] = 1
            
        ans += 3
    
    # print(new_wm_data.shape)
    ## row 15, 16
    feature = wm_data_np[5]
    for j in range(17):
        if getIndex(unique_feat_lis[-1], feature[j]) == 0:
            new_wm_data[16, j] = 1
        elif getIndex(unique_feat_lis[-1], feature[j]) == 1:
            new_wm_data[15, j] = 1
    
    ## row 17, 18
    new_wm_data = new_wm_data.T
    wm_data_np = wm_data_np.T
    for k in range(17):
        for i in range(17, 19):
            j = i - 11 ## 6-7
            new_wm_data[k, i] = wm_data_np[k, j]

    new_label = np.zeros((17, 1), dtype = int)
    new_label = wm_data_np[:, -1]
    # print(new_label)

    return new_wm_data, new_label

