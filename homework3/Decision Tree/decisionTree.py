from getWMData import getWMData
import numpy as np 

def splitFeatures(dataset, axis, value):
    # features = dataset[:, 1:8]
    features = dataset

    new_dataset = []

    for feature_vector in features:
        feature_vector = feature_vector.tolist()
        if feature_vector[axis] == value:
            new_feature_vector = feature_vector[0:axis]
            new_feature_vector.extend(feature_vector[axis + 1:])
            new_dataset.append(new_feature_vector)

    return np.array(new_dataset)

import collections

def getMaxFeatures(label_lis):
    label_cnt = collections.defaultdict(int)

    for i in label_lis:
        label_cnt[i] += 1

    ## sort
    new_label_cnt = sorted(label_cnt.items(), reverse = True)

    return new_label_cnt[0][0]

def getShannonEntropy(dataset):
    # dataset = dataset[:, 1:8]
    num_data = len(dataset)

    label_cnt = collections.defaultdict(int)

    for feature_vector in dataset:
        label = feature_vector[-1]

        if label not in label_cnt.keys():
            label_cnt[label] = 0

        label_cnt[label] += 1

    shannon_entropy = 0.0

    import math 

    for key in label_cnt:
        prob = float(label_cnt[key]) / num_data

        shannon_entropy -= prob * math.log(prob, 2) 
    
    return shannon_entropy


def getBestFeature(dataset, features):
    # features = dataset[:, 7]
    # features = ['色泽', '根蒂', '敲击', '纹理', '脐部', '触感']
    # print(features)
    # print(type(dataset))
    # dataset = dataset[:, 1:8]
    dataset = np.asarray(dataset)
    # print(dataset)
    

    num_feature = dataset.shape[1] - 1
    # print(num_feature)

    shannon_entropy = getShannonEntropy(dataset = dataset)

    max_infor_gain = 0.0
    best_feature = -1

    for i in range(num_feature):
        feature_lis = []

        for data in dataset:
            feature_lis.append(data[i])
        # print(feature_lis)
        
        unique_feature = set(feature_lis)

        new_entropy = 0.0

        for feature in unique_feature:
            new_dataset = splitFeatures(dataset = dataset, axis = i, value = feature)

            probability = len(new_dataset) / float(len(dataset))


            new_entropy += probability * getShannonEntropy(new_dataset)

        
        infor_gain = shannon_entropy - new_entropy

        if infor_gain > max_infor_gain:
            max_infor_gain = infor_gain
            best_feature = i

    return best_feature



def judgeSameFeatures(dataset):
    dataset = dataset[:, 1:8]
    dataset = np.asarray(dataset)
    num_feature = dataset.shape[1] - 1
    num_data = dataset.shape[0]

    first_feature = ''
    flag = True

    for i in range(num_feature):
        first_feature = dataset[0][i]

        for j in range(1, num_data):
            if first_feature != dataset[j][i]:
                return False
    
    return flag

def decisionTree(origin_dataset, features):
    # origin_dataset = np.asarray(origin_dataset)
    # dataset = origin_dataset[:, 1:8]

    dataset = np.asarray(origin_dataset)

    labels = []
    for i in dataset:
        labels.append(i[-1])
    
    if labels.count(labels[0]) == len(labels):
        return labels[0]

    if judgeSameFeatures(dataset) or len(dataset[0]) == 1:
        return getMaxFeatures(label_lis = labels)

    best_feature = getBestFeature(dataset, features)
    

    best_feature_name = features[best_feature]

    # print(best_feature_name, best_feature)

    tree = {best_feature_name : {}}

    del(features[best_feature])
    # print(dataset)
    values = []
    for i in dataset:
        values.append(i[best_feature])
    # print(values)

    unique_val = set(values)
    # print(unique_val)

    for val in unique_val:
        new_features = features[:]
        
        new_tree = decisionTree(splitFeatures(dataset = dataset, axis = best_feature, value = val), new_features)

        tree[best_feature_name][val] = new_tree

    return tree

def train(dataset, features):
    dataset = np.asarray(dataset)
    dataset = dataset[:, 1:8]
    tree = decisionTree(dataset, features)
    return tree


def getIndex(data_vector, target):

    for i in range(len(data_vector)):
        if data_vector[i] == target:
            break
    
    return i

def checkValueType(dic):
    flag = True
    for key, val in dic.items():
        if isinstance(val, str) == False:
            flag = False
            break
    return flag

def test(origin_test_data, tree):
    # print(tree.keys())
    # print(tree[list(tree)[0]])
    origin_test_data = np.asarray(origin_test_data)
    test_data = origin_test_data[:, 1:7]
    test_label = origin_test_data[:, 7]
    # print(test_data)
    # print(test_label)
    features = ['色泽', '根蒂', '敲击', '纹理', '脐部', '触感']

    # current_feature_idx = 0
    current_vector_idx = -1
    correct = 0.0
    tree_cpy = tree

    for feature_vector in test_data:
        current_vector_idx += 1
        flag = False
        tree_cpy = tree
        # tree_cpy = tree_cpy[list(tree_cpy)[0]]
        ans = 0
        while flag == False:
            test_current_feature = feature_vector[getIndex(features, list(tree_cpy)[0])]
            # print(test_current_feature) ## e.g. '青绿'
            # if ans == 0:
            # print(tree_cpy, 'aaaaa')
            if isinstance(tree_cpy, dict) == False:
                ## not a tree, but a str
                if tree_cpy == test_label[current_vector_idx]:
                    correct += 1
                    flag = True
                continue
            tree_cpy = tree_cpy[list(tree_cpy)[0]]


            for key, value in tree_cpy.items():

                if str(key) == str(test_current_feature):
                    if tree_cpy[key] == '是' or tree_cpy[key] == '否':
                        flag = True
                        if tree_cpy[key] == test_label[current_vector_idx]:
                            correct += 1
                        # current_vector_idx += 1
                        break

                    else:
                                    # current_feature_idx = getIndex(features, list(tree_cpy[key])[0])
                        tree_cpy = tree_cpy[key]
                        # print(tree_cpy, 'bbbbbb')
                        ans += 1
                        break
            
    correct /= test_data.shape[0]

    return correct
        

    # ans = -1
    # correct = 0.0
    # current_feature = ''
    # for test_vector in test_data:
    #     ans += 1
    #     label = test_label[ans]
    #     while True:
    #         for i in features:
    #             if i == str(list(tree)[0]):
    #                 current_feature = i
    #                 print(current_feature, 'aaaaa')
    #                 break
    #                 # features.remove(current_feature)
    #         # if isinstance(tree[list(tree)[0]], dict) == False:
    #         if checkValueType(tree) == True:
    #             ## not a dict
    #             # print(type(tree[list(tree)[0]]))
    #             # print(tree[list(tree)[0]], 'mmmm')
    #             # print(label, 'aaaa')
    #             # print(test_vector[getIndex(test_vector, current_feature)], 'bbbb')
    #             # print(current_feature)
                
    #             if test_vector[getIndex(test_vector, current_feature)] == label:
    #                 correct += 1
    #             break

    #         else:
    #             # tree = tree[list(tree)[0]]


    # correct /= test_data.shape[0]
    # print(correct)
    # return correct

    

if __name__ == "__main__":
    train_data, test_data = getWMData()
    features = ['色泽', '根蒂', '敲击', '纹理', '脐部', '触感']

    tree = train(train_data, features)
    
    print('Decision tree is:')
    print(tree)
    acc = test(test_data, tree)
    print('Testing accuracy is:', '\t', acc)


