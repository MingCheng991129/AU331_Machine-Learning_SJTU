import numpy as np 

class threeLayersNN():
    def __init__(self, input_, hidden, output, learning_rate):
        self.input_ = input_
        self.hidden = hidden
        self.output = output
        self.learning_rate = learning_rate

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def MSELoss(self, ground_truth, predict):
        ## the input are c*1 array
        row = predict.shape[0]
        temp = np.zeros((row, 1), dtype = int)
        idx = np.argmax(predict)
        temp[idx] = 1 ## set one at corressponding position
        ## compute the loss using temp and ground_truth
        loss = (1 / 2) * (predict-temp) ** 2
        MSELoss = np.sum(loss, axis=0) ## sum through the whole vector
        return MSELoss

    def initialize(self):
        ## input_ is integer (D)
        ## hidden is an integer (H)
        ## weight1 is a matrix (d * hidden)
        ## output is a vector (c*1)
        D = self.input_
        H = self.hidden
        C = self.output
        weight1 = np.random.uniform(0, 1, (D, H)) ## initialize randomly
        weight2 = np.random.uniform(0, 1, (H, C)) ## initialize randomly
        bias1 = np.full((H, 1), 0.1) ## fill with 0.1
        bias2 = np.full((C, 1), 0.1) ## fill with 0.1

        return D, H, C, weight1, weight2, bias1, bias2


    def forward(self, train_data, ground_truth, weight1, weight2, bias1, bias2):

        a1 = train_data ## d*1
        z1 = np.dot(weight1.T, a1) + bias1 ## h*1
        a2 = self.sigmoid(z1) ## h*1
        z2 = np.dot(weight2.T, a2) + bias2 ## c*1
        out = np.argmax(z2, axis=0)
        MSELoss = self.MSELoss(ground_truth, z2)
    
        return MSELoss, out, weight1, weight2, bias1, bias2, a1, a2, z1, z2

    def backpropagation(self, train_data, ground_truth, weight1, weight2, bias1, bias2):
        ## back propagation
        MSELoss, out, weight1, weight2, bias1, bias2, a1, a2, z1, z2 = self.forward(train_data, ground_truth, weight1, weight2, bias1, bias2)
        ground_truth_new = np.zeros((3,1),dtype=int)
        ground_truth_new[ground_truth] += 1
        dw2 = np.dot((z2 - ground_truth_new),a2.T).T ## h*c
        db2 = (z2 - ground_truth_new) ## c*1


        dL_dz2 = z2 - ground_truth_new ## c*1
        dw1_temp1 = np.dot(weight2, dL_dz2) ## h*c*c*1 = h*1
        dw1_temp2 = np.dot(dw1_temp1, self.sigmoid(z1).T) ## h*1*1*h = h*h
        dw1_temp3 = np.dot(dw1_temp2, (1 - self.sigmoid(z1))) ## h*h*h*1 = h*1
        dw1 = np.dot(dw1_temp3, a1.T).T ## (h*1*1*d).T = d*h

        db1 = dw1_temp3

        weight1 -= self.learning_rate * dw1
        weight2 -= self.learning_rate * dw2
        bias1 -= self.learning_rate * db1
        bias2 -= self.learning_rate * db2
        ## MSELoss and out are not changed
        return weight1, weight2, bias1, bias2, MSELoss, out













