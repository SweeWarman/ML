import numpy as np
from abc import ABC,abstractmethod
from functools import reduce

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
        self.gradientsU = None
        self.weightedInput = None

    def __call__(self,u):
        return self.forwardpass(u)

    def setup(self):
        if self.prevLayer is not None:
            self.numInputs = self.prevLayer.totalNodes + 1
            self.weights = self.weightInit(self.numOutputs,self.numInputs)
            self.input       = np.zeros((self.numInputs,1))
            self.gradientsW  = np.ones((self.numOutputs,self.numInputs))
            self.gradientsU  = np.ones((self.numOutputs,self.numInputs))
        else:
            self.gradientsW = np.zeros((1,self.numInputs))
            self.gradientsU = np.zeros((1,self.numInputs))

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
            weightedInput = np.mean(self.weightedInput,axis=1).reshape(self.totalNodes,1)
            outputs       = np.mean(self.outputs,axis=1).reshape(self.totalNodes,1)
            input         = np.mean(self.input,axis=1).reshape(self.numInputs,1)
            self.gradientsW *= np.dot(self.Node.gradient(weightedInput,outputs),
                                    np.transpose(input))
            self.gradientsU *= np.dot(self.Node.gradient(weightedInput,outputs),
                                    np.ones((1,self.numInputs)))*self.weights
            self.gradientsU[:,-1] = 0 # last column is the bias term
            if self.nextLayer is not None:
                self.accumgrad = np.dot(self.nextLayer.gradientsU[:,:-1].sum(axis=0).reshape(self.numOutputs,1),
                                        np.ones((1,self.numInputs)))
                self.gradientsW *= self.accumgrad
                self.gradientsU *= self.accumgrad

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
        dJdy = costgrad(ypred,yactual)
        self.SetCostGradient(dJdy)
        for i in range(self.totalLayers):
            self.layers[self.totalLayers - 1 - i].backwardpass()

    def SetCostGradient(self,dJdy):
        gradientU = np.dot(dJdy,np.ones((1,self.layers[-1].numInputs)))
        self.layers[-1].gradientsU = gradientU.copy()
        self.layers[-1].gradientsW = gradientU.copy()


def MeanSquaredError(ypred,yactual):
    n = ypred.shape[1]
    mse = 1/(0.5*n)*np.sum(np.linalg.norm(ypred-yactual,axis=1)**2)
    return mse

def SquaredErrorGrad(ypred,yactual):
    msegrad = np.mean(ypred-yactual,axis=1)
    return msegrad.reshape(ypred.shape[0],1)

def TestSetEvaluation(nn,testData,testLabel):
    total = testData.shape[1]
    ypred = np.argmax(nn(testData),axis=0)
    yactual = np.argmax(testLabel,axis=0)
    n = len([abs(val) for val in ypred-yactual if abs(val) < 1])
    print(n,"/",total)


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

            # Set cost gradient
            dJdy = grad(ypred,yactual)
            network.SetCostGradient(dJdy)

            # backward pass
            network.backwardpass(ypred,yactual,grad)

            # update weights by gradient descent
            for layer in network.layers[1:]:
                layer.weights -= learningRate*layer.gradientsW

        if testData is not None and testLabel is not None:
            TestSetEvaluation(network,testData,testLabel)
        