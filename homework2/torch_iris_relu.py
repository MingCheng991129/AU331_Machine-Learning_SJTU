import torch as torch
import numpy as np 
import matplotlib.pyplot as plt 
import scipy.io as sio
from sklearn.datasets import load_iris
import torch.nn as nn
import torch.nn.functional as F

def getData():
        ### load iris data
    # iris_data = np.loadtxt('iris.data.txt')
    iris_data = load_iris()
    # print(iris_data['data'])
    # print(iris_data['target'])

    data = iris_data['data']
    labels = iris_data['target']

    # print(data.shape)
    # print(labels.shape)

    ### get random data
    idx = list(range(data.shape[0]))
    np.random.shuffle(idx)
    # print(idx)
    data = data[idx]
    labels = labels[idx]
    # print(data)
    # print(labels)

    ### normalize
    data /= np.max(abs(data))
    data -= np.mean(data, axis= 0)
    # print(data)

    train_num = int(data.shape[0] * 0.8)
    test_num = data.shape[0] - train_num

    train_data = data[0:train_num]
    test_data = data[train_num:]

    train_label = labels[0:train_num]
    test_label = labels[train_num:]

    return train_data, test_data, train_label, test_label, data, labels, train_num

train_data, test_data, train_label, test_label, data, labels, train_num = getData()

Hidden = 100
N = train_num
Input = data.shape[1]
Output = labels.shape[0]
learning_rate = 0.01


class threeLayersNN(nn.Module):
    def __init__(self):
        super(threeLayersNN, self).__init__()
        self.fc1 = nn.Linear(Input, Hidden)
        self.fc3 = nn.Linear(Hidden, Output)

    def forward(self, X):
        X = self.fc1(X)
        X = F.relu(X)
        X = self.fc3(X)
        X = F.softmax(X)
        return X


from torch.autograd import Variable
net = threeLayersNN()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)


def train(num_epoch):
    loss_lis = []  ## train loss
    epoch_lis = [] ## epoch
    acc_lis_train = [] ## train acc
    loss_lis_test = [] ## test loss
    acc_lis_test = [] ## test acc
    X = Variable(torch.Tensor(train_data).float())
    Y = Variable(torch.Tensor(train_label).long())

    PATH = 'C:/Users/1/Desktop/iris_sigmoid_save/'

    for epoch in range(num_epoch):
        optimizer.zero_grad()
        out = net(X)
        _, pred_train = torch.max(out, 1)
        Y_np_train = Y.numpy().tolist()
        pred_np_train = pred_train.numpy().tolist()
        acc_train = 0
        for i in range(len(Y_np_train)):
            if Y_np_train[i] == pred_np_train[i]:
                acc_train += 1
        acc_train /= train_num
        acc_train *= 100

        loss = criterion(out, Y)
        loss.backward()
        optimizer.step()

        test_loss, test_acc = test()

        if epoch % 10 == 0:
            epoch_lis.append(epoch)
            loss_lis.append(loss.item())
            loss_lis_test.append(test_loss)
            acc_lis_test.append(test_acc)
            acc_lis_train.append(acc_train)
        

        torch.save(net.state_dict(), PATH + str(epoch))
            
        print('epoch:', epoch, 'training loss:', '%0.6f'%loss.item(), 'training accuracy:', '%0.3f'%acc_train, \
            '\t', 'testing loss:', '%0.6f'%test_loss, 'testing accuracy:', '%0.3f'%test_acc)

    print('final accuracy %.3f %%' % acc_lis_test[-1])

    return epoch_lis, loss_lis, acc_lis_train, loss_lis_test, acc_lis_test

def test():

    X = Variable(torch.Tensor(test_data).float())
    Y = torch.Tensor(test_label).long()
    out = net(X)
    loss = criterion(out, Y)
    _, y_pred = torch.max(out, 1)
    Y_np = Y.numpy().tolist()
    y_pred_np = y_pred.numpy().tolist()
    # print(Y_np, y_pred_np)
    acc = 0
    for i in range(len(Y_np)):
        if Y_np[i] == y_pred_np[i]:
            acc += 1

    acc /= (data.shape[0] - train_num)
    acc *= 100

    return loss.item(), acc


def plot(num_epoch):
    epoch_lis, loss_lis, acc_lis_train, loss_lis_test, acc_lis_test = train(num_epoch)
    plt.figure(1)
    plt.title('training & testing loss with epoch')
    plt.plot(epoch_lis, loss_lis, label = 'training loss')
    plt.plot(epoch_lis, loss_lis_test, label = 'testing loss')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('loss')

    plt.figure(2)
    plt.title('training & testing accuracy with epoch')
    plt.plot(epoch_lis, acc_lis_train, label = 'training accuracy')
    plt.plot(epoch_lis, acc_lis_test, label = 'testing accuracy')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('accuracy')

    plt.show()

    

if __name__ == "__main__":
    plot(num_epoch = 500)
    




