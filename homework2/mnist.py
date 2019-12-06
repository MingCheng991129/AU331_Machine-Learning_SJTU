import torch
import torchvision
import numpy as np
from torch import nn
from torch.nn import init
from torch.autograd import Variable
from torch.utils import data
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torchvision.transforms as transforms

BATCH_SIZE = 100
LEARNING_RATE = 1e-3

def getData():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(3,1,1)),    ### debug
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    train_data = torchvision.datasets.MNIST(root = './mnist', train = True, \
        transform = transform, download = False)
    test_data = torchvision.datasets.MNIST(root = './mnists', train = False, \
        transform = transform, download = False)
    
    from torch.utils.data.sampler import SubsetRandomSampler
    num_train = len(train_data)

    ## divide the dataset into train
    train_sample = SubsetRandomSampler(list(range(num_train))[:])

    train_loader = data.DataLoader(train_data, batch_size = BATCH_SIZE, sampler = train_sample, num_workers = 0)
    test_loader = data.DataLoader(test_data, batch_size = BATCH_SIZE, num_workers = 0)
    ### show an example
    # print(train_data.train_data.size())                 
    # print(train_data.train_labels.size()) 
    # plt.imshow(train_data.train_data[0].numpy(), cmap='gray')
    # plt.title('%i' % train_data.train_labels[0])
    # plt.show()

    return train_loader, test_loader


class myCNN(nn.Module):
    def __init__(self):
        super(myCNN, self).__init__()

        ### input 3 * 28 * 28 
        self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 16, kernel_size = 5, stride = 1, padding = 1)
        self.norm1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, 5, 1, 2)
        self.norm2 = nn.BatchNorm2d(32)
        self.maxpool = nn.MaxPool2d(2, 2)
        #self.fc1 = nn.Linear(64, 500)
        self.fc1 = nn.Linear(1152, 500)
        self.fc2 = nn.Linear(500, 10)
        self.dropout = nn.Dropout(p = 0.4)
    
    def forward(self, x):
        x = x.cuda()
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.maxpool(F.relu(x))
        x = x.cuda()
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.maxpool(F.relu(x))
        x = x.cuda()
        #x = x.view(-1, 7 * 7 * 32)
        x = x.view(x.size(0), -1)
        x = x.cuda()
        # print(x.size(0))
        x = self.dropout(x)
        x = x.cuda()
        x = F.relu(self.fc1(x))
        x = x.cuda()
        x = self.dropout(x)
        x = x.cuda()
        x = self.fc2(x)
        x = x.cuda()
        x = F.softmax(x)

        return x

net = myCNN()

device = torch.device("cuda:0")
# net.to(device)
net = net.cuda(device = device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr = LEARNING_RATE)

def train(num_epoch):
    train_loader, test_loader = getData()
    train_loss_lis = []   
    train_acc_lis = []
    test_loss_lis = []
    test_acc_lis = []
    '''''
    training for num_epoch times
    each time calculate train_loss, train_acc, test_loss, test_acc
    '''''
    PATH = 'C:/Users/1/Desktop/ML/hw2/save2/'
    for epoch in range(num_epoch):
        
        train_loss = 0
        net.train()
        correct_num = 0

        for img, label in train_loader:
            ## train_loader: 60000 / batch_size per time
            
            img = img.cuda(device = device)
            label = label.cuda(device = device)

            optimizer.zero_grad()
            predict = net(img) ## size: batch_size * 10  
            loss = criterion(predict, label) ## tensor(0.9415, device='cuda:0', grad_fn=<NllLossBackward>)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * img.shape[0] ## change the scale of train_loss

            _, new_predict = torch.max(predict, dim=1) ## maximum of every row   (100*1)

            ## label.data: 1*100
            new_res = new_predict.eq(label.data.view_as(new_predict)) ## same dimension, find equal
            res = np.squeeze(new_res.cpu().numpy())
            ## res: 100*1  [0,0,1,1...]
        
            correct_num += np.sum(res)   ## in batch_size

        net.eval()


        ## deal with train_loss and accuracy
        len_train_loader = len(train_loader.dataset)

        train_loss /= len_train_loader
        train_acc = correct_num / len_train_loader

        train_loss_lis.append(train_loss)
        train_acc_lis.append(train_acc)

        test_loss, test_acc = test()
        test_loss_lis.append(test_loss)
        test_acc_lis.append(test_acc)

        torch.save(net.state_dict(), PATH + str(epoch))

        print('epoch:', epoch, 'training loss:', '%0.6f'%train_loss, 'training accuracy:', '%0.4f'%train_acc, \
            '\t', 'testing loss:', '%0.6f'%test_loss, 'testing accuracy:', '%0.4f'%test_acc)


    return train_loss_lis, train_acc_lis, test_loss_lis, test_acc_lis

def test():
    train_loader, test_loader = getData()
    test_loss = 0
    net.eval()

    correct_lis = []
    all_lis = []

    test_loss_lis = []
    test_acc_lis = []
    correct_num = 0


    for img, label in test_loader:
        # img.to(device)
        # label.to(device)
        img = img.cuda(device = device)
        label = label.cuda(device = device)

        pred = net(img)

        loss = criterion(pred, label)
        
        test_loss += loss.item() * img.shape[0]

        _, new_pred = torch.max(pred, dim = 1)
        #print(type(label))
        new_res = new_pred.eq(label.data.view_as(new_pred))

        res = np.squeeze(new_res.cpu().numpy())
        correct_num += np.sum(res)

    len_test_loader = len(test_loader.dataset)
    test_loss /= len_test_loader
    test_acc = correct_num / len_test_loader
    # test_loss_lis.append(test_loss)
    # test_acc_lis.append(test_acc)

    return test_loss, test_acc


def plot(train_loss_lis, train_acc_lis_epoch, test_loss_lis, test_acc_lis, num_epoch):

    print('final accuracy:', '%0.4f'%np.mean(test_acc_lis))
    x = []
   
    for i in range(num_epoch):
        x.append(i)
    
    plt.figure(1)
    plt.plot(x, train_loss_lis, label = 'training loss')
    plt.plot(x, test_loss_lis, label = 'testing loss')
    plt.title("training & testing loss with epoch")
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('loss')

    plt.figure(2)
    plt.plot(x, train_acc_lis, label = 'training accuracy')
    plt.plot(x, test_acc_lis, label = 'testing accuracy')
    plt.title('training & testing accuracy with epoch')
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    
    plt.show()


if __name__ == "__main__":
    getData()

    train_loss_lis, train_acc_lis, test_loss_lis, test_acc_lis = train(50)
   
    plot(train_loss_lis, train_acc_lis, test_loss_lis, test_acc_lis, 50)

