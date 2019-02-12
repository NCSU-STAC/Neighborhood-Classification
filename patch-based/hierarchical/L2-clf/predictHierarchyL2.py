import numpy as np
from tflearn.data_utils import shuffle
import slumnet as net
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import pickle

trainX = np.load('/pvfs2/kgadira@oss-storage-0-108/pvfs2/kgadira/data/filtered-3p-tr-X')
trainY = np.load('/pvfs2/kgadira@oss-storage-0-108/pvfs2/kgadira/data/filtered-3p-tr-Y')
print trainX.shape, trainY.shape
train_6p_Y = np.load('/pvfs2/kgadira@oss-storage-0-108/pvfs2/kgadira/data/filtered-6p-tr-Y') 
test_6p_Y = np.load('/pvfs2/kgadira@oss-storage-0-108/pvfs2/kgadira/data/filtered-6p-te-Y') 
print train_6p_Y.shape, test_6p_Y.shape


# Read files 
testX = np.load('/pvfs2/kgadira@oss-storage-0-108/pvfs2/kgadira/data/filtered-3p-te-X')
testY = np.load('/pvfs2/kgadira@oss-storage-0-108/pvfs2/kgadira/data/filtered-3p-te-Y')
print testY.shape
final_all_X = np.vstack((trainX, testX))
final_all_Y_labels = np.hstack((trainY, testY)) 
#final_all_Y_labels = np.argmax(final_all_Y_one_Hot, axis = 1)
final_all_6p_Y = np.hstack((train_6p_Y, test_6p_Y))

#final_all_Y_splits = np.hstack((final_all_Y_labels, final_all_3p_Y, final_all_6p_Y))
#print final_all_Y_splits.shape

print final_all_X.shape, final_all_Y_labels.shape
print np.max(final_all_Y_labels)
path_to_model = '../../models/filtered-model-3-cl-final/final-model.tflearn'
model = net.model
model.load(path_to_model)

actual_labels = testY
result_labels = np.zeros((testY.shape[0],)) - 1
for i in range(testX.shape[0]):
    curr_X = testX[i,:,:,:]
    #print curr_X.shape
    curr_X = curr_X.reshape([-1, 40,40,18])
    result_prob = model.predict(curr_X)[0]
    result_labels[i] = np.argmax(result_prob)



acc = accuracy_score(actual_labels, result_labels)
cm = confusion_matrix(actual_labels, result_labels)
cr = classification_report(actual_labels,result_labels, target_names=['Informal','Formal'])
print 'Test accuracy = {}\n'.format(acc)
#print 'Slum accuracy = {}\n'.format(cm[0,0]/np.sum(cm[0,:]))
print 'Confusion Matrix \n {}\n'.format(cm)
print 'Classification Report \n {}\n'.format(cr)

#print final_all_X.shape, final_all_Y.shape
#predict for test batch
#final_all_X = final_all_X.reshape([-1, 40, 40, 18])
actual_labels = final_all_Y_labels
result_labels = np.zeros((final_all_X.shape[0],)) - 1
for i in range(final_all_X.shape[0]):
    curr_X = final_all_X[i,:,:,:]
    #print curr_X.shape
    curr_X = curr_X.reshape([-1, 40,40,18])
    result_prob = model.predict(curr_X)[0]
    result_labels[i] = np.argmax(result_prob)



acc = accuracy_score(actual_labels, result_labels)
cm = confusion_matrix(actual_labels, result_labels)
cr = classification_report(actual_labels,result_labels, target_names=['Informal','Formal'])
print 'Overall accuracy = {}\n'.format(acc)
#print 'Slum accuracy = {}\n'.format(cm[0,0]/np.sum(cm[0,:]))
print 'Confusion Matrix \n {}\n'.format(cm)
print 'Classification Report \n {}\n'.format(cr)


final_all_out_X = final_all_X[result_labels == actual_labels,: ,: ,:] 
final_all_out_Y_labels = final_all_Y_labels[result_labels == actual_labels]
final_all_out_6p_Y_labels = final_all_6p_Y[result_labels == actual_labels]

print final_all_out_X.shape, final_all_out_Y_labels.shape, final_all_out_6p_Y_labels.shape
# Split into training and test again
# Count training and testing to perform splits
train_proportion = 0.7

#extract only the data relevant to correctly classified instances for class informal
k = 0
final_6cl_X = final_all_out_X[final_all_out_Y_labels == k,:,:,:]
final_6cl_Y = final_all_out_6p_Y_labels[final_all_out_Y_labels == k]

'''
n_vals = curr_class_indices.shape[0]
np.random.shuffle(curr_class_indices)
n_train = int(train_proportion * n_vals)
print curr_class_indices.shape
curr_train_indices = curr_class_indices[0:n_train]
curr_test_indices = curr_class_indices[n_train:]
curr_train_X = final_all_out_X[curr_train_indices, :,:,:]
curr_train_Y = final_all_out_Y_labels[curr_train_indices]
curr_test_X = final_all_out_X[curr_test_indices, :,:,:]
current_test_Y = final_all_out_Y_labels[curr_test_indices]
train_6cl_Y = final_all_out_6p_Y_labels[curr_train_indices]
test_6cl_Y = final_all_out_6p_Y_labels[curr_test_indices]
'''


train_6cl_X, test_6cl_X, train_6cl_Y, test_6cl_Y = train_test_split(final_6cl_X, final_6cl_Y, train_size = 0.7, stratify = final_6cl_Y)

print train_6cl_X.shape, train_6cl_Y.shape, test_6cl_X.shape, test_6cl_Y.shape

for k in range(4):
    print 'Level 3 split TRAIN, number of elements of class {} = {}'.format(k, np.sum(train_6cl_Y==k))
    print 'Level 3 split TEST, number of elements of class {} = {}'.format(k, np.sum(test_6cl_Y==k))


f = open('/pvfs2/kgadira@oss-storage-0-108/pvfs2/kgadira/data/filtered-6p-patch-tr-X','w')
pickle.dump(train_6cl_X,f)
f.close()


f = open('/pvfs2/kgadira@oss-storage-0-108/pvfs2/kgadira/data/filtered-6p-patch-tr-Y','w')
pickle.dump(train_6cl_Y,f)
f.close()

f = open('/pvfs2/kgadira@oss-storage-0-108/pvfs2/kgadira/data/filtered-6p-patch-te-X','w')
pickle.dump(test_6cl_X,f)
f.close()

f = open('/pvfs2/kgadira@oss-storage-0-108/pvfs2/kgadira/data/filtered-6p-patch-te-Y','w')
pickle.dump(test_6cl_Y,f)
f.close()
