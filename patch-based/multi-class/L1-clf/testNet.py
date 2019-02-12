import numpy as np
from tflearn.data_utils import shuffle
import buildingnet as net
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

# Read files 
testX = np.load('/home/kgadira/data/final-pt-te-2-X')
testY = np.load('/home/kgadira/data/final-pt-te-2-Y')

path_to_model = '../models/model-2-cl-final/final-model.tflearn'
model = net.model
model.load(path_to_model)

print testX.shape, testY.shape
#predict for test batch
testX = testX.reshape([-1, 40, 40, 18])
result_probs = model.predict(testX)

result_labels = np.argmax(result_probs, axis = 1)

actual_labels = np.argmax(testY, axis = 1)
#actual_labels = testY
print result_probs.shape, result_labels.shape 

acc = accuracy_score(actual_labels, result_labels)
cm = confusion_matrix(actual_labels, result_labels)
cr = classification_report(actual_labels,result_labels, target_names=['Urban','Other'])
print 'Overall accuracy = {}\n'.format(acc)
#print 'Slum accuracy = {}\n'.format(cm[0,0]/np.sum(cm[0,:]))
print 'Confusion Matrix \n {}\n'.format(cm)
print 'Classification Report \n {}\n'.format(cr)

for i in range(actual_labels.shape[0]):
    print 'Actual = {}, predicted = {}'.format(actual_labels[i], result_labels[i])
