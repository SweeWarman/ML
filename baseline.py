'''
nnRef = Ref.Network([numInputs,15,10])

nnRef.weights[0] = nn.layers[1].weights[:,:-1]
nnRef.biases[0][:,0]  = nn.layers[1].weights[:,-1]
nnRef.weights[1] = nn.layers[2].weights[:,:-1]
nnRef.biases[1][:,0] = nn.layers[2].weights[:,-1]

ui = miniD[:,0].reshape(numInputs,1)

myy = nn(ui)
refy = nnRef.feedforward(ui)

yi = miniL[:,0].reshape(10,1)
dJdy = SquaredErrorGrad(myy,yi)
nn.SetCostGradient(dJdy)
nn.backwardpass(myy)

db,dw = nnRef.backprop(ui,yi)

print(nn.layers[-2].gradientsW[5,:])
print(dw[-2][5,:])

print("error=",np.linalg.norm(myy - refy))
'''

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

ypred = nn(testData[:,:2])
yp0 = ypred1[:,:2]
l0  = testLabel[:,:2]
nn.backwardpass(yp0,l0,SquaredErrorGrad)
grad3 = nn.layers[-1].gradientsW[0,:-1]
print(sum(grad1))


gradW,gradB = nnRef.update_mini_batch(testDataSet[:2],0.01)

print(sum(gradW[-1][0,:]))

