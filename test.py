import numpy as np
from ReadDataset import GetTrainingData
from NN import (Network, MeanSquaredError,
                SquaredErrorGrad, Sigmoid,
                Linear, Train)

import BaseNN as Ref

data,label = GetTrainingData()

N = 50000 
trainData,trainLabel = data[:,0:N], label[:,0:N]
testData,testLabel = data[:,N:], label[:,N:]

numInputs = data.shape[0]

nn = Network()
nn.AddLayer(numInputs,Linear)
nn.AddLayer(100,Sigmoid)
nn.AddLayer(100,Sigmoid)
nn.AddLayer(10,Sigmoid)
nn.setup()


Train(nn,MeanSquaredError,SquaredErrorGrad,trainData,trainLabel,100,1.0,50,testData,testLabel)

