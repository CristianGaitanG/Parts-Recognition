import neurolab as nl
import numpy as np
import pylab as pl

error = []        #In this we store error value of each iteration or epoach
i=[0,0,1,1]
j=[0,1,0,1]
k=[[i]+[j]]
print k
inp=[[0,0],[0,1],[1,0],[1,1]]
output=np.array([0,1,1,0])
size = len(output)
output=output.reshape(size,1)
print inp
print output
# Create network with 1 hidden layer and random initialized
#nl.net.newff() is feed forward neural network
#1st argument is min max values of predictor variables
#2nd argument is no.of nodes in each layer i.e 4 in hidden 1 in o/p
#transf is transfer function applied in each layer
net = nl.net.newff([[0, 1],[0,1]],[4,1],transf=[nl.trans.LogSig()] * 2)
net.trainf = nl.train.train_rprop
# Training network
#net.train outputs error which is appended to error variable
error.append(net.train(inp,output, show=0, epochs = 100,goal=0.001))
#plotting epoches Vs error
#we can use this plot to specify the no.of epoaches in training to reduce time
pl.figure(1)
pl.plot(error[0])
pl.xlabel('Number of epochs')
pl.ylabel('Training error')
pl.grid()
pl.show()
# Simulate network(predicting)
predicted_values = net.sim([[0,1]])
#converting predicted values into classes by using threshold
predicted_class=predicted_values
predicted_class[predicted_values>0.5]=1
predicted_class[predicted_values<=0.5]=0
#predicted classes
print predicted_class
