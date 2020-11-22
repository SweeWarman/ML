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
nn.AddLayer(15,Sigmoid)
nn.AddLayer(15,Sigmoid)
nn.AddLayer(10,Sigmoid)
nn.setup()

nnRef = Ref.Network([numInputs,15,15,10])

nnRef.weights[0] = nn.layers[1].weights[:,:-1].copy()
nnRef.biases[0][:,0]  = nn.layers[1].weights[:,-1].copy()
nnRef.weights[1] = nn.layers[2].weights[:,:-1].copy()
nnRef.biases[1][:,0] = nn.layers[2].weights[:,-1].copy()
nnRef.weights[2] = nn.layers[3].weights[:,:-1].copy()
nnRef.biases[2][:,0] = nn.layers[3].weights[:,-1].copy()

trainDataSet = [(x.reshape(numInputs,1),y.reshape(10,1)) for x,y in zip(trainData.T,trainLabel.T)]
testDataSet = [(x.reshape(numInputs,1),y.reshape(10,1)) for x,y in zip(testData.T,testLabel.T)]


# don't forget to uncomment shuffling
#nnRef.SGD(trainDataSet,100,100,0.01,testDataSet)

#Train(nn,MeanSquaredError,SquaredErrorGrad,trainData,trainLabel,100,0.01,100,testData,testLabel)

ypred2 = nnRef.feedforward(testDataSet[0][0])
gradB0,gradW0 = nnRef.backprop(testDataSet[0][0],testDataSet[0][1])

#print(sum(gradW0[-1][0,:]))
#print(sum(gradW0[-2][0,:]))

ypred1 = nn(testData[:,0].reshape(numInputs,1))
yp0 = ypred1[:,0].reshape(10,1)
l0  = testLabel[:,0].reshape(10,1)
nn.backwardpass(yp0,l0,SquaredErrorGrad)
grad1 = nn.layers[-1].gradientsW[0,:-1]
print(sum(grad1))

ypred1 = nn(testData[:,1].reshape(numInputs,1))
yp0 = ypred1[:,0].reshape(10,1)
l0  = testLabel[:,0].reshape(10,1)
nn.backwardpass(yp0,l0,SquaredErrorGrad)
grad2 = nn.layers[-1].gradientsW[0,:-1]
print(sum(grad1))

#ypred = nn(testData[:,:2])
#yp0 = ypred1[:,:2]
#l0  = testLabel[:,:2]
#nn.backwardpass(yp0,l0,SquaredErrorGrad)
#grad3 = nn.layers[-1].gradientsW[0,:-1]
#print(sum(grad1))


gradW,gradB = nnRef.update_mini_batch(testDataSet[:1],0.01)

print(sum(gradW[-1][0,:]))

"""
weightsGrad = [np.zeros(l.weights.shape) for l in nn.layers[1:]]
for k in range(10):
    # forward pass
    ypred = nn(trainData[:,k].reshape(numInputs,1))
    ypred0 = nnRef.feedforward(trainDataSet[k][0])

    # backward pass
    nn.backwardpass(ypred,trainLabel[:,k].reshape(10,1),SquaredErrorGrad)
    gradb,gradW = nnRef.backprop(trainDataSet[k][0],trainDataSet[k][1])

    weightsGrad = [wg+l.gradientsW for wg,l in zip(weightsGrad,nn.layers[1:])]

weightsGrad = [w/10 for w in weightsGrad]
gradW,gradB = nnRef.update_mini_batch(trainDataSet[:10],0.01)
"""