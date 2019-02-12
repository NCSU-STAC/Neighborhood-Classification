import numpy as np
from tflearn.data_utils import shuffle
import simplenet as net

# Read files 
'''
trainX = np.load('/scratch/slums/bl-slums/gt/final-pt-tr-2-X')
trainY = np.load('/scratch/slums/bl-slums/gt/final-pt-tr-2-Y')
testX = np.load('/scratch/slums/bl-slums/gt/final-pt-te-2-X')
testY = np.load('/scratch/slums/bl-slums/gt/final-pt-te-2-Y')

'''
trainX = np.load('/scratch/slums/bl-slums/gt/filtered-L3-6p-tr-X')
trainY = np.load('/scratch/slums/bl-slums/gt/filtered-L3-6p-tr-Y')
testX = np.load('/scratch/slums/bl-slums/gt/filtered-L3-6p-te-X')
testY = np.load('/scratch/slums/bl-slums/gt/filtered-L3-6p-te-Y')

print trainX.shape, trainY.shape, testX.shape, testY.shape

print np.max(trainY), np.max(testY)


# One - Hot encode trainY
for i in range(trainY.shape[0]):
    curr = np.zeros((1,4))
    curr[0,trainY[i]] = 1
    if i == 0:
       final_train_Y = curr
    else:
       final_train_Y = np.vstack((final_train_Y, curr))
print final_train_Y.shape

trainY = final_train_Y


model = net.model

trainX = trainX.reshape([-1, 40, 40, 18])
print trainX.shape

# Shuffle data
trainX, trainY = shuffle(trainX, trainY)

# Train the model
model.fit(trainX, trainY, n_epoch=100,validation_set= 0.2, show_metric=True, run_id="deep_nn1", batch_size = 128,snapshot_step=100000)
model.save("/home/kgadira/NN-Slum/filtered-model-6-cl-final/final-model.tflearn")

