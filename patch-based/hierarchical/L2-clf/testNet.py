import numpy as np
from tflearn.data_utils import shuffle
import slumnet as net
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

# Read files 
testX = np.load('/home/kgadira/data/filtered-3p-te-X')
testY = np.load('/home/kgadira/data/filtered-3p-te-Y')

path_to_model = '../../models/filtered-model-3-cl-final/final-model.tflearn'
model = net.model
model.load(path_to_model)

testY = testY.reshape((testY.shape[0],1))
print testX.shape, testY.shape

#predict for test batch
testX = testX.reshape([-1, 40, 40, 18])
result_probs = model.predict(testX)

result_labels = np.argmax(result_probs, axis = 1)

#actual_labels = np.argmax(testY, axis = 1)
actual_labels = testY
print result_probs.shape, result_labels.shape 


acc = accuracy_score(actual_labels, result_labels)
cm = confusion_matrix(actual_labels, result_labels)
cr = classification_report(actual_labels,result_labels, target_names=['Informal','Formal'])
print 'Overall accuracy = {}\n'.format(acc)
#print 'Slum accuracy = {}\n'.format(cm[0,0]/np.sum(cm[0,:]))
print 'Confusion Matrix \n {}\n'.format(cm)
print 'Classification Report \n {}\n'.format(cr)

 
