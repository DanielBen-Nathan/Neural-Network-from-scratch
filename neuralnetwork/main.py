import numpy as np
from keras.datasets import mnist
import NeuralNetwork as nn
import matplotlib.pyplot as plt


def formatY(y):
    newy = np.zeros((y.shape[0], 10))
    for i in range(y.shape[0]):
        zeros = np.zeros((1, 10))
        zeros[0][y[i]] = 1
        newy[i] = zeros
    return newy

def formatX(X):
    X = X / 255
    X = np.reshape(X, (X.shape[0], 28 * 28))
    return X

if __name__ == "__main__":
    (X, y),(tX,ty)=mnist.load_data()

    y=formatY(y)
    X=formatX(X)

    NN = nn.NeuralNetwork(28*28,28*28,10)
    print("Initial Cost: " + str(NN.cost(X,y)))
    NN.train(X,y,20,0.0001,True,True)

    ty=formatY(ty)
    tX=formatX(tX)

    test_cost , test_acc = NN.validation(tX,ty)
    print("Validation Cost: " + str(test_cost))
    print("Validation Accuracy: " + str(test_acc))

    indexTest = 3
    plt.imshow(np.reshape(tX[indexTest]*255,(28,28)),cmap=plt.cm.binary)
    print("prediction for image is: "+str(np.argmax(NN.forwardProp(tX[indexTest]))))
    plt.show()
