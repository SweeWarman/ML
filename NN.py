import numpy as np
from abc import ABC,abstractmethod
from functools import reduce
import time

class Node(ABC):
    @abstractmethod
    def activate(self,u):
        pass

    @abstractmethod
    def gradient(self,x,y):
        pass

class Linear(Node):
    def activate(self,u):
        return u

    def gradient(self,x,y):
        return 1

class Sigmoid(Node):
    def activate(self,u):
        return 1/(1+np.exp(-u))

    def gradient(self,x,y):
        return y*(1-y)

class Layer:
    def __init__(self,totalNodes,Node,weightInit=np.random.rand):
        self.Node       = Node()
        self.nodes      = []
        self.totalNodes = totalNodes
        self.prevLayer  = None
        self.nextLayer  = None
        self.weights    = None
        self.outputs    = None
        self.input      = None
        self.numInputs  = 0
        self.numOutputs = totalNodes
        self.weightInit = weightInit
        self.gradientsW = None
        self.propgrad   = None
        self.weightedInput = None

    def __call__(self,u):
        return self.forwardpass(u)

    def setup(self):
        if self.prevLayer is not None:
            self.numInputs = self.prevLayer.totalNodes + 1
            self.weights = self.weightInit(self.numOutputs,self.numInputs)
            self.input       = np.zeros((self.numInputs,1))
            self.gradientsW  = np.ones((self.numOutputs,self.numInputs))
        else:
            self.gradientsW = np.zeros((1,self.numInputs))

    def initializeGrad(self,dataSize=1):
        if dataSize == 1:
            self.gradientsW  = np.ones((self.numOutputs,self.numInputs))
            self.propgrad    = np.zeros((self.numOutputs,1))
        else:
            self.gradientsW  = np.ones((dataSize,self.numOutputs,self.numInputs))
            self.propgrad    = np.zeros((self.numOutputs,dataSize))

    def forwardpass(self,u):
        self.input  = np.vstack((u,np.ones((1,u.shape[1]))))
        if self.weights is None:
            self.outputs = u
            return self.outputs
        self.weightedInput = np.dot(self.weights,self.input)
        self.outputs = self.Node.activate(self.weightedInput)
        return self.outputs

    def backwardpass(self):
        if self.prevLayer is not None:
            for i in range(self.outputs.shape[1]):
                self.propgrad[:,[i]] = self.propgrad[:,[i]]*self.Node.gradient(self.weightedInput[:,[i]],self.outputs[:,[i]])
                self.gradientsW[i] = np.dot(self.propgrad[:,[i]],self.input[:,[i]].T)
                # Avoid this computation for the input layer
                if self.prevLayer.prevLayer is not None:
                    self.prevLayer.propgrad[:,[i]] = np.dot(np.transpose(self.weights[:,:-1]),self.propgrad[:,[i]])

        return self.gradientsW

class Network:
    def __init__(self):
        self.layers = []
        self.totalLayers = 0
        self.numInputs = 0
        self.numOutputs = 0

    def __call__(self,u):
        return self.forwardpass(u)

    def AddLayer(self,numNodes,nodeType,weightInit=np.random.randn):
        newLayer = Layer(numNodes,nodeType,weightInit)
        if len(self.layers) > 0:
            self.layers[-1].nextLayer = newLayer
            newLayer.prevLayer = self.layers[-1]
        self.layers.append(newLayer)
        self.totalLayers += 1

    def setup(self):
        for l in self.layers:
            l.setup()
        self.numInputs = self.layers[0].totalNodes
        self.numOutputs = self.layers[-1].totalNodes

    def forwardpass(self,u):
        y = lambda x,f:f(x)
        return reduce(y,self.layers,u)

    def backwardpass(self,ypred,yactual,costgrad):
        dataSize = ypred.shape[1]
        dJdy = costgrad(ypred,yactual)

        # set the dimensions of weight gradient matrices
        for i in range(1,self.totalLayers):
            self.layers[i].initializeGrad(dataSize)

        self.SetCostGradient(dJdy)
        for i in range(self.totalLayers):
            self.layers[self.totalLayers - 1 - i].backwardpass()

    def SetCostGradient(self,dJdy):
        self.layers[-1].propgrad = dJdy


def MeanSquaredError(ypred,yactual):
    n = ypred.shape[1]
    mse = 1/(0.5*n)*np.sum(np.linalg.norm(ypred-yactual,axis=1)**2)
    return mse

def SquaredErrorGrad(ypred,yactual):
    msegrad = ypred-yactual
    return msegrad

def TestSetEvaluation(nn,testData,testLabel):
    ypred = np.argmax(nn(testData),axis=0)
    yactual = np.argmax(testLabel,axis=0)
    n = len([abs(val) for val in ypred-yactual if abs(val) < 1])
    return n


def Train(network,cost,grad,data,label,epochs,learningRate,minibatch = 25,testData=None,testLabel=None):
    totalSize = data.shape[1]
    indices = [i for i in range(totalSize)]

    for i in range(epochs):
        # sample mini batch from data
        np.random.shuffle(indices)
        dataS = data[:,indices]
        labelS = label[:,indices]
        numMiniBatches = int(totalSize/minibatch)
        for j in range(numMiniBatches):
            start = j*minibatch
            end   = start + minibatch
            sampleData = dataS[:,start:end]
            yactual = labelS[:,start:end]

            # forward pass
            ypred = network(sampleData)
            # backward pass

            network.backwardpass(ypred,yactual,grad)

            for layer in network.layers[1:]:
                layer.weights -= learningRate*np.mean(layer.gradientsW,axis=0)


        if testData is not None and testLabel is not None:
            n = TestSetEvaluation(network,testData,testLabel)
            print("epoch: %d, %d/%d"%(i,n,testLabel.shape[1]))

def ShowTestData(i,network,testData):
    from matplotlib import pyplot as plt
    img = testData[:,[i]].reshape(28,28)
    print(np.argmax(network(testData[:,[i]])))
    plt.imshow(img)
    plt.show()

        