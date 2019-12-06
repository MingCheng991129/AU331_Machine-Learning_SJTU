import np_getdata
import np_model
import numpy as np
import matplotlib.pyplot as plt 

train_data, test_data, train_label, test_label, data, labels, train_num = np_getdata.getData()

net = np_model.threeLayersNN(input_=train_data.shape[1], hidden=10, output=3, learning_rate=1e-2)
_, _, _, weight1, weight2, bias1, bias2 = net.initialize()

def train_iris(num_epoch):
    train_loss_lis = []
    train_acc_lis = []
    test_loss_lis = []
    test_acc_lis = []
    PATH1_ = './weight1/'
    PATH2_ = './weight2/'
    PATH3_ = './bias1/'
    PATH4_ = './bias2/'
    for epoch in range(num_epoch):
        PATH1 = PATH1_ + str(epoch)
        PATH2 = PATH2_ + str(epoch)
        PATH3 = PATH3_ + str(epoch)
        PATH4 = PATH4_ + str(epoch)
        f1 = open(PATH1, 'w')
        f2 = open(PATH2, 'w')
        f3 = open(PATH3, 'w')
        f4 = open(PATH4, 'w')

        train_loss = train_acc = 0
        for row in range(train_num):
            if row == 0:
                new_weight1 = weight1
                new_weight2 = weight2
                new_bias1 = bias1
                new_bias2 = bias2

            train_data_new = train_data[row,:].reshape(-1, 1)
            ground_truth = train_label[row]

            #MSELoss, out, _, _, _, _, _, _, _, _ = net.forward(train_data_new, ground_truth, new_weight1, new_weight2, new_bias1, new_bias2)
            weight1_, weight2_, bias1_, bias2_, MSELoss, out = net.backpropagation(train_data_new, ground_truth, new_weight1, new_weight2, new_bias1, new_bias2)
            
            
            new_weight1 = weight1_
            new_weight2 = weight2_
            new_bias1 = bias1_
            new_bias2 = bias2_
            train_loss += MSELoss
            if out == ground_truth:
                train_acc += 1

        f1.write(str(weight1_))
        f1.write('\n')
        f2.write(str(weight2_))
        f2.write('\n')
        f3.write(str(bias1_))
        f3.write('\n')
        f4.write(str(bias2_))
        f4.write('\n')
        f1.close()
        f2.close()
        f3.close()
        f4.close()

        train_acc /= train_num
        train_loss /= train_num

        train_loss_lis.append(train_loss)
        train_acc_lis.append(train_acc)

        test_acc, test_loss = test_iris(new_weight1, new_weight2, new_bias1, new_bias2)
        test_acc_lis.append(test_acc)
        test_loss_lis.append(test_loss)

        print('epoch:', epoch, 'training loss:', '%0.6f'%train_loss, 'training accuracy:', '%0.3f'%train_acc, \
            '\t', 'testing loss:', '%0.6f'%test_loss, 'testing accuracy:', '%0.3f'%test_acc)
    
    print('final accuracy:', '%0.3f'%test_acc_lis[-1])
    return train_loss_lis, train_acc_lis, test_loss_lis, test_acc_lis

def test_iris(test_weight1, test_weight2, test_bias1, test_bias2):
    test_num = data.shape[0] - train_num
    test_loss = test_acc = 0
    for row in range(test_num):
        test_data_new = test_data[row,:].reshape(-1, 1)
        test_ground_truth = test_label[row]
        MSELoss, out, _, _, _, _, _, _, _, _,  = net.forward(test_data_new, test_ground_truth, test_weight1, test_weight2, test_bias1, test_bias2)

        test_loss += MSELoss
        if out == test_ground_truth:
            test_acc += 1

    test_acc /= test_num
    test_loss /= test_num

    return test_acc, test_loss


def plot(train_loss_lis, train_acc_lis, test_loss_lis, test_acc_lis, num_epoch):
    epoch = []
    for i in range(num_epoch):
        epoch.append(i)

    plt.figure(1)
    plt.plot(epoch, train_loss_lis, label = 'training loss')
    plt.plot(epoch, test_loss_lis, label = 'testing loss')
    plt.title('training and testing loss with epoch')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('loss')

    plt.figure(2)
    plt.plot(epoch, train_acc_lis, label = 'training accuracy')
    plt.plot(epoch, test_acc_lis, label = 'testing accuracy')
    plt.title('training and testing accuracy with epoch')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('accuracy')

    plt.show()


if __name__ == "__main__":
    train_loss_lis, train_acc_lis, test_loss_lis, test_acc_lis = train_iris(1000)
    plot(train_loss_lis, train_acc_lis, test_loss_lis, test_acc_lis, 1000)

