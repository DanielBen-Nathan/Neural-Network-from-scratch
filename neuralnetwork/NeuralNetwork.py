import numpy as np
import matplotlib.pyplot as plt


# finish: gradient checking,  variable layers, biases, regularisation, save/load, speed up calc, wrong but works?, visualise layers

class NeuralNetwork(object):

    def __init__(self, inputSize, hiddenSize, outputSize):
        self.inputLayerSize = inputSize
        self.hiddenLayerSize = hiddenSize
        self.outputLayerSize = outputSize

        self.w1 = np.random.randn(self.inputLayerSize, self.hiddenLayerSize)
        self.w2 = np.random.randn(self.hiddenLayerSize, self.outputLayerSize)

    def sigmoid(self, z):

        return (1 / (1 + np.exp(-z)))

    def sigmoidDeriv(self, z):
        return np.multiply(self.sigmoid(z), (1 - self.sigmoid(z)))

    def cost(self, X, y):
        m = y.shape[0]
        J = (1 / (2 * m)) * np.sum((y - self.forwardProp(X)) ** 2)

        # J = (1 / (m)) * np.sum( np.sum(  (np.multiply(y,np.log(self.forwardProp(X)))) - np.multiply((1-y),np.log(1-self.forwardProp(X)) )))
        return J

    def forwardProp(self, X):
        # self.a1 = np.insert(X, 0, 1, axis=1)
        self.a1 = X
        self.z2 = np.dot(self.a1, self.w1)
        self.a2 = self.sigmoid(self.z2)
        # self.a2 = np.insert(self.a2, 0, 1, axis=1)
        self.z3 = np.dot(self.a2, self.w2)
        self.a3 = self.sigmoid(self.z3)
        return self.a3

    def backProp(self, X, y):

        m = y.shape[0]
        '''
        w1grad = np.zeros(self.w1.shape);
        w2grad = np.zeros(self.w2.shape);
        for t in range(m):
            #pass
            delta3 = self.a3[t,:] - y[t,:]
            z2deriv = self.sigmoidDeriv(self.z2[t,:])
            z2deriv = np.insert(z2deriv, 0, 1, axis=0)

            z2deriv= np.reshape(z2deriv,(785,1))
            delta2 = np.multiply((self.w2*delta3),self.sigmoidDeriv(z2deriv))
            w2grad+=delta3.T*self.a2[t,:]
            w1grad+=delta2[2:]*self.a1[t,:]

        #z3 = np.insert(self.z3, 0, 1, axis=1)
        delta3 = np.multiply(-(y-self.forwardProp(X)),self.sigmoidDeriv(self.z3))
        dJdW2 = np.dot(self.a2.T, delta3)
        #z2 = np.insert(self.z2, 0, 1, axis=1)
        delta2 = np.dot(delta3, self.w2.T) * self.sigmoidDeriv(self.z2)
        dJdW1 = np.dot(X.T, delta2)
        '''
        self.forwardProp(X)
        dJdW2 = np.dot(self.a2.T, (-(y - self.a3) * self.sigmoidDeriv(self.z3)))
        dJdW1 = np.dot(self.a1.T, (np.dot(-(y - self.a3) * self.sigmoidDeriv(self.a3), self.w2.T) * self.sigmoidDeriv(self.z2)))

        # delta3 = np.multiply(-(y - self.a3), self.sigmoidDeriv(self.z3))
        # dJdW2 = np.dot(self.a2.T, delta3)

        # delta2 = np.dot(delta3, self.w2.T) * self.sigmoidDeriv(self.z2)
        # dJdW1 = np.dot(self.a1.T, delta2)

        return dJdW1, dJdW2

    def train(self, X, y, iterations, lr, graph=False,trackAccuracy=False):
        costs = []
        accuracies = []
        for i in range(iterations):
            print("Iteration: " + str(i + 1))
            dJdW1, dJdW2 = self.backProp(X, y)
            self.w1 -= lr * dJdW1
            self.w2 -= lr * dJdW2
            J = self.cost(X, y)
            print("Cost: " + str(J))
            if(trackAccuracy):
                acc = self.accuracy(X,y)
                print("Accuracy: " + str(acc))
                accuracies.append(acc)
            costs.append(J)
            print()

        if (graph):
            plt.figure(1)
            plt.plot(range(len(costs)), costs)
            plt.grid(True)
            plt.xlabel("Iteration")
            plt.ylabel("Cost")
            if(trackAccuracy):
                plt.figure(2)
                plt.plot(range(len(accuracies)), accuracies)
                plt.grid(True)
                plt.xlabel("Iteration")
                plt.ylabel("Accuracy")


            plt.show()



    def validation(self, tX, ty):

        J = self.cost(tX, ty)
        acc = self.accuracy(tX, ty)
        return J, acc

    def accuracy(self, X, y):
        m = y.shape[0]
        sum = 0
        for i in range(m):
            if (np.argmax(y[i, :]) == np.argmax(self.forwardProp(X[i]))):
                sum += 1
        acc = sum / m
        return acc

    def gradientChecking(self):
        pass