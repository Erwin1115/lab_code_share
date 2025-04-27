from mxnet.gluon import data as gdata
import sys
import time
from mxnet import nd
import numpy as np
from DDHIP import *

mnist_train = gdata.vision.FashionMNIST(train=False)

mnist_test = gdata.vision.FashionMNIST(train=False)

batch_size = 10

transformer = gdata.vision.transforms.ToTensor()

train_iter = gdata.DataLoader(mnist_train.transform_first(transformer
                                                          ),
                              batch_size, shuffle=False)

test_iter = gdata.DataLoader(mnist_test.transform_first(transformer),
                             batch_size, shuffle=False)

num_inputs = 784

num_hiddens = 30

num_outputs = 10

W1 = nd.random.normal(scale=0.01, shape=(num_inputs, num_hiddens))

W2 = nd.random.normal(scale=0.01, shape=(num_hiddens, num_outputs))

params = [W1, W2]



def sigmoid(X):

    return 1 / (1 + nd.exp(-X))


def sigmoid_prime(y):

    return sigmoid(y) * (1 - sigmoid(y))


class net:

    def __init__(self, params, batch_size, lr):

        self.W1 = params[0]

        self.W2 = params[1]

        self.batch_size = batch_size

        self.lr = lr

        setup = DDHIP_Setup(l=num_inputs)

        self.mpk, self.msk = setup.setup()

    def forward(self, X):

        X = X.reshape((-1, num_inputs))
        # print(Dot(X.asnumpy(), X.asnumpy().T).dot())
        # Z = nd.array(Dot(X, self.W1).dot())
        # start1 = time.time()

        A = nd.random.uniform(shape=(X.shape[1],X.shape[1]))

        A_1 = nd.linalg_inverse(A)

        w1 = nd.dot(X, A_1)

        w2 = nd.dot(A, self.W1)

        res = nd.zeros(shape=(w1.shape[0], w2.shape[1]))
        # print(res.shape)

        for i in range(w1.shape[0]):

            m=0

            for j in range(w2.shape[1]):

                start3 = time.time()

                encrypt = DDHIP_Encrypt(w1[i, :], self.mpk, self.msk)

                ct = encrypt.encrypt()
                end3 = time.time()

                m = m+1

                print(m)

                # print('本地加密时间', end3 - start3)

                decrypt = DDHIP_Decrypt(w2[:, j], self.msk, ct)

                res[i, j] = decrypt.decrypt()
        Z = res


        self.h = sigmoid(Z)

        o = nd.dot(self.h, self.W2)

        return self.softmax2(o)

    def evaluate_accuracy(self, data_iter, net):

        acc_sum, n = 0.0, 0

        for X, y in data_iter:

            y = y.astype('float32')

            acc_sum += (net(X).argmax(axis=1) == y).sum().asscalar()

            n += y.size

        return acc_sum / n

    def softmax2(self, X):

        row_max = X.max(axis=1)

        row_max = row_max.reshape(-1, 1)

        X = X - row_max

        X_exp = X.exp()

        partition = X_exp.sum(axis=1, keepdims=True)

        return X_exp / partition

    def cross_entropy(self, y_hat, y):

        return -nd.pick(y_hat, y).log()  #

    def backword(self, dLdo, X):

        X = X.reshape((-1, num_inputs))

        print('X1=', X.shape)

        self.W2_grad = nd.dot(self.h.T, dLdo)

        dLdh = nd.dot(dLdo, self.W2.T)
        dLdz = dLdh * sigmoid_prime(self.h)

        # print(dLdz.shape)
        A = nd.random.uniform(shape=(X.shape[0], X.shape[0]))

        print(A.shape)

        A_1 = nd.linalg_inverse(A)

        X = nd.dot(X.T, A_1)

        print('X2=', X.shape)

        dLdz1 = nd.dot(A, dLdz)

        print('dldz', dLdz1.shape)

        res = nd.zeros(shape=(X.shape[0], dLdz1.shape[1]))

        # print(res.shape)
        for i in range(X.shape[0]):

            for j in range(dLdz1.shape[1]):

                encrypt = DDHIP_Encrypt(X[i, :], self.mpk, self.msk)

                ct = encrypt.encrypt()

                decrypt = DDHIP_Decrypt(dLdz1[:, j], self.msk, ct)

                res[i, j] = decrypt.decrypt()

        self.W1_grad = res
        # self.W1_grad = nd.dot(X, dLdz1)

    def sgd(self, batch_size):

        self.W2 -= self.lr * (self.W2_grad / self.batch_size)

        self.W1 -= self.lr * (self.W1_grad / self.batch_size)

    def train(self, train_iter, test_iter, num_epochs, batch_size):

        a = 0
        for epoch in range(num_epochs):

            b = 0

            i=0

            train_l_sum, train_acc_sum, n, start = 0.0, 0.0, 0, time.time()

            for X, y in train_iter:

                start2 = time.time()

                print(a, b, len(train_iter))

                y_hat = self.forward(X)

                # print(nd.pick(self.softmax2(y_hat),y),y)

                l = self.cross_entropy(y_hat, y).sum()  #

                y_eye = nd.eye(10)[y]  #

                self.backword(y_hat - y_eye, X)  #

                self.sgd(batch_size)

                end2 = time.time()

                print('time:=', end2-start2)

                y = y.astype('float32')

                train_l_sum += l.asscalar()

                i += 1

                print(i)

                train_acc_sum += (self.forward(X).argmax(axis=1) == y).sum().asscalar()

                n += y.size

                # break
            test_acc = self.evaluate_accuracy(test_iter, self.forward)

            print(

                f'epoch {epoch + 1},loss {round(train_l_sum / n, 4)},train_accuracy {round(train_acc_sum / n, 4)},test_accuracy {test_acc} time {round(time.time() - start, 4)}')


num_epochs, lr = 5, 0.5

net = net(params, batch_size, lr)

net.train(train_iter, test_iter, num_epochs, batch_size)
