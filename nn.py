import numpy as np
import pickle
import random


class NN():
    def __init__(self):
        # load params if there are in file
        # otherwise initialize random params and save them to file 
        try:
            with open('parameters.pickle', 'rb') as f:
                param = pickle.load(f)

            self.W1 = param[0]
            self.b1 = param[1]
            self.W2 = param[2]
            self.b2 = param[3]
            self.W3 = param[4]
            self.b3 = param[5]
        except:
            self.init_params()


    def init_params(self):
        self.W1 = np.random.rand(30, 784) - 0.5
        self.b1 = np.random.rand(30, 1) - 0.5
        self.W2 = np.random.rand(15, 30) - 0.5
        self.b2 = np.random.rand(15, 1) - 0.5
        self.W3 = np.random.rand(10, 15) - 0.5
        self.b3 = np.random.rand(10, 1) - 0.5

        self.save_params()


    def save_params(self):
        data_to_save = [self.W1, self.b1, self.W2, self.b2, self.W3, self.b3]

        #save to file
        with open('parameters.pickle' , 'wb') as f:
            pickle.dump(data_to_save, f)



    def forward_prop(self, X):
        self.Z1 = self.W1.dot(X) + self.b1
        self.A1 = ReLU(self.Z1)
        self.Z2 = self.W2.dot(self.A1) + self.b2
        self.A2 = ReLU(self.Z2)
        self.Z3 = self.W3.dot(self.A2) + self.b3
        self.A3 = softmax(self.Z3)
        return self.A3



    def back_prop(self, X, Y, m, alpha):
        one_hot_Y = one_hot(Y)
        dZ3 = self.A3 - one_hot_Y
        dW3 = 1/m * dZ3.dot(self.A2.T)
        db3 = 1/m * np.sum(dZ3)

        dZ2 = self.W3.T.dot(dZ3) * ReLU_deriv(self.Z2)
        dW2 = 1/m * dZ2.dot(self.A1.T)
        db2 = 1/m * np.sum(dZ2)

        dZ1 = self.W2.T.dot(dZ2) * ReLU_deriv(self.Z1)
        dW1 = 1/m * dZ1.dot(X.T)
        db1 = 1/m * np.sum(dZ1)

        # update params
        self.W1 = self.W1 - alpha * dW1
        self.b1 = self.b1 - alpha * db1

        self.W2 = self.W2 - alpha * dW2
        self.b2 = self.b2 - alpha * db2

        self.W3 = self.W3 - alpha * dW3
        self.b3 = self.b3 - alpha * db3



    def train(self, X, Y, start_alpha, end_alpha,  iterations, package_size):
        m = X.shape[1]
        alpha = start_alpha

        #devide data to smaler packages
        packages_X = []
        packages_Y = []
        for i in range(package_size, m, package_size):
            packages_X.append(X[:, i-package_size:i])
            packages_Y.append(Y[i-package_size:i])

        for i in range(1, iterations + 1):
            #shuffle packages
            temp = list(zip(packages_X, packages_Y))
            random.shuffle(temp)
            res1, res2 = zip(*temp)
            packages_X, packages_Y = list(res1), list(res2)

            for package_idx in range(len(packages_X)):  
                x = packages_X[package_idx]
                y = packages_Y[package_idx] 
                self.forward_prop(x)
                self.back_prop(x, y, x.shape[1], alpha)
            if i % 10 == 0:
                print("Iteration: ", i)
                self.forward_prop(X)
                predictions = get_predictions(self.A3)
                print(get_accuracy(predictions, Y))
                print('alfa: ', alpha)
                alpha = (end_alpha - start_alpha)/(iterations + 1) * i + start_alpha

        
    def make_predictions(self, X):
        self.forward_prop(X)
        predictions = get_predictions(self.A3)
        return predictions
    






#utils
def ReLU(Z):
    return np.maximum(Z, 0)


def softmax(Z):
    Z -= np.max(Z, axis=0)
    A = np.exp(Z) / np.sum(np.exp(Z), axis=0)
    return A


def ReLU_deriv(Z):
    return Z > 0


def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, 10))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y


def get_predictions(A3):
    return np.argmax(A3, 0)


def get_accuracy(predictions, Y):
    return np.sum(predictions == Y) / Y.size