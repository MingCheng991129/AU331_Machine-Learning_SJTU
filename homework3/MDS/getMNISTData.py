import pandas as pd 
import numpy as np 
import random

def img2csv(img_data, label, out, num):
    img = open(img_data, 'rb')
    label = open(label, 'rb')

    img.read(16)
    label.read(8)

    img_lis = []

    for i in range(num):
        temp = [ord(label.read(1))]

        for j in range(28 * 28):
            temp.append(ord(img.read(1)))

        img_lis.append(temp)

    out = open(out, 'w')

    for i in img_lis:
        ## write data
        out.write(','.join(str(pix) for pix in i) + '\n')

    img.close()
    label.close()
    out.close()

    return out

   

def loadData(filename):
    data = pd.read_csv(filename, names = range(785))
    # print(data.describe())
    # print(data)

    ## get label == 1 and 2
    data1 = data[data[0] == 1]
    data2 = data[data[0] == 2]
    # print(data1)
    # print(data2)
    # print(data1.describe())
    # print(data2.describe())

    data12 = pd.concat([data1, data2], axis = 0)
    # print(data12.describe())
    data12 = data12.sample(frac=1).reset_index(drop = True)

    dataset = data12.iloc[:, 1:] / float(255)
    label = data12.iloc[:, 0]

    label = np.array(label)
    dataset = np.array(dataset)

    return dataset, label

