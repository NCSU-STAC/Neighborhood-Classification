'''
Filter training + test data from L2 to get only correctly classified Slum Instances

'''


from osgeo import gdal
import ogr
import numpy as np
import fiona
import xgboost
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
import pickle

# Set seed
np.random.seed(100)

# Read files 
trainX = np.load('/pvfs2/kgadira@oss-storage-0-108/pvfs2/kgadira/data/filtered-L1-L2-tr-X')
trainY = np.load('/pvfs2/kgadira@oss-storage-0-108/pvfs2/kgadira/data/filtered-L1-L2-tr-Y')
testX = np.load('/pvfs2/kgadira@oss-storage-0-108/pvfs2/kgadira/data/filtered-L1-L2-te-X')
testY = np.load('/pvfs2/kgadira@oss-storage-0-108/pvfs2/kgadira/data/filtered-L1-L2-te-Y')


train_L3_Y = np.load('/pvfs2/kgadira@oss-storage-0-108/pvfs2/kgadira/data/filtered-L1-L3-tr-Y')
test_L3_Y = np.load('/pvfs2/kgadira@oss-storage-0-108/pvfs2/kgadira/data/filtered-L1-L2-te-Y')


trainY = trainY.ravel()
testY = testY.ravel()
train_L3_Y = train_L3_Y.ravel()
test_L3_Y = test_L3_Y.ravel()


models_list = ['xgboost','rf','dt','gnb','knn','mlp','adaboost']
#trainX = np.nan_to_num(trainX)
#testX = np.nan_to_num(testX)
for clf_path in models_list:

    print "==================================================================="
    
    model = pickle.load(open('./models/filtered-L2-{}-model.sav'.format(clf_path),'rb'))
    result = model.predict(trainX)
    acc = accuracy_score(trainY, result)
    cm = confusion_matrix(trainY, result)
    cr = classification_report(trainY,result, target_names=['Informal','Formal'])
    print '{} Classifier training results:'.format(clf_path)
    print 'Overall accuracy = {}\n'.format(acc)
    #print 'Slum accuracy = {}\n'.format(cm[0,0]/np.sum(cm[0,:]))
    print 'Confusion Matrix \n {}\n'.format(cm)
    print 'Classification Report \n {}\n'.format(cr)


    result = model.predict(testX)
    acc = accuracy_score(testY, result)
    cm = confusion_matrix(testY, result)
    cr = classification_report(testY,result, target_names=['Informal','Formal'])
    print '{} CLassifier test results:'.format(clf_path)
    print 'Overall accuracy = {}\n'.format(acc)
    #print 'Slum accuracy = {}\n'.format(cm[0,0]/np.sum(cm[0,:]))
    print 'Confusion Matrix \n {}\n'.format(cm)
    print 'Classification Report \n {}\n'.format(cr)

    print "==================================================================="
    
    

# filter correctly classified training and test data
final_all_data_X = np.vstack((trainX, testX))
final_all_data_Y = np.hstack((trainY, testY))

#print 'L2 data shape training: {}, L2 data shape testing: {}'.format(trainY.shape, testY.shape)
#print 'L3 data shape training: {}, L3 data shape testing: {}'.format(train_L3_Y.shape, test_L3_Y.shape)
final_all_data_L3_Y = np.hstack((train_L3_Y, test_L3_Y))

clf_path='rf'
model = pickle.load(open('./models/filtered-L2-{}-model.sav'.format(clf_path),'rb'))
result = model.predict(final_all_data_X)
    
print final_all_data_X.shape
result = model.predict(final_all_data_X)
final_all_out_X = final_all_data_X[result == final_all_data_Y,:] 
final_all_out_Y = final_all_data_Y[result == final_all_data_Y]
final_all_out_L3_Y = final_all_data_L3_Y[result == final_all_data_Y]


## Split the correctly classified instances into training and testing
train_proportion = 0.7

#extract only the data relevant to correctly classified instances for class informal
k = 0
final_all_out_Y_final = final_all_out_Y[final_all_out_Y == k]
final_all_out_X_final = final_all_out_X[final_all_out_Y == k,:]
final_all_out_L3_Y_final = final_all_out_L3_Y[final_all_out_Y == k]

print 'L3 data Min index = {}, Max Index = {}'.format(np.min(final_all_out_L3_Y_final), np.max(final_all_out_L3_Y_final)) 

for i in range(4):
    print 'Class {}: Number of values = {}'.format(i, np.sum(final_all_out_L3_Y_final==i))
#n_vals = curr_class_indices.shape[0]
#np.random.shuffle(curr_class_indices)
#n_train = int(train_proportion * n_vals)
#print curr_class_indices.shape
#curr_train_indices = curr_class_indices[0:n_train]
#curr_test_indices = curr_class_indices[n_train:]
#curr_train_X = final_all_out_X[curr_train_indices, :]
#curr_train_Y = final_all_out_Y_labels[curr_train_indices]
#curr_test_X = final_all_out_X[curr_test_indices, :]
#curr_test_Y = final_all_out_Y_labels[curr_test_indices]
#train_L3_Y = final_all_out_L3_Y[curr_train_indices]
#test_L3_Y = final_all_out_L3_Y[curr_test_indices]

curr_train_X, curr_test_X, train_L3_Y, test_L3_Y = train_test_split(final_all_out_X_final, final_all_out_L3_Y_final, train_size = 0.7, stratify = final_all_out_L3_Y_final)

print 'Train X shape = {}, train y shape = {}, testX shape = {}, testY shape = {}'.format(curr_train_X.shape, train_L3_Y.shape, curr_test_X.shape, test_L3_Y.shape) 

for i in range(4):
    print 'Class {}: Number of train values = {}, Number of test values = {}'.format(i, np.sum(train_L3_Y==i), np.sum(test_L3_Y==i))


f = open('/home/kgadira/data/filtered-L2-L3-tr-X','w')
pickle.dump(curr_train_X,f)
f.close()

f = open('/home/kgadira/data/filtered-L2-L3-te-X','w')
pickle.dump(curr_test_X,f)
f.close()

f = open('/home/kgadira/data/filtered-L2-L3-tr-Y','w')
pickle.dump(train_L3_Y,f)
f.close()

f = open('/home/kgadira/data/filtered-L2-L3-te-Y','w')
pickle.dump(test_L3_Y,f)
f.close()

