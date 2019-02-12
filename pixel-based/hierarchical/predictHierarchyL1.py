'''
Filter training + test data from L1 to get only correctly classified Urban Instances

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
import pickle

# Set seed
np.random.seed(100)

# Read files 
trainX = np.load('/home/kgadira/data/final-px-tr-2-Xa')
trainY = np.load('/home/kgadira/data/final-px-tr-2-Ya')
testX = np.load('/home/kgadira/data/final-px-te-2-Xa')
testY = np.load('/home/kgadira/data/final-px-te-2-Ya')

train_L2_Y = np.load('/home/kgadira/data/final-px-tr-3-Ya')
test_L2_Y = np.load('/home/kgadira/data/final-px-te-3-Ya')

train_L3_Y = np.load('/home/kgadira/data/final-px-tr-6-Ya')
test_L3_Y = np.load('/home/kgadira/data/final-px-te-6-Ya')

train_L2_X = np.load('/home/kgadira/data/final-px-tr-3-Xa')
train_L3_X = np.load('/home/kgadira/data/final-px-tr-6-Xa')


trainY = trainY.ravel()
testY = testY.ravel()
train_L2_Y = train_L2_Y.ravel()
test_L2_Y = test_L2_Y.ravel()
train_L3_Y = train_L3_Y.ravel()
test_L3_Y = test_L3_Y.ravel()

print trainX.shape, trainY.shape, testX.shape, testY.shape, train_L2_X.shape, train_L2_Y.shape, train_L3_Y.shape

models_list = ['xgboost','rf','dt','gnb','knn','mlp','adaboost']
#trainX = np.nan_to_num(trainX)
#testX = np.nan_to_num(testX)
for clf_path in models_list:

    print "==================================================================="
    
    model = pickle.load(open('../models/final-L1-{}-model.sav'.format(clf_path),'rb'))
    result = model.predict(trainX)
    acc = accuracy_score(trainY, result)
    cm = confusion_matrix(trainY, result)
    cr = classification_report(trainY,result, target_names=['Urban','Background'])
    print '{} Classifier training results:'.format(clf_path)
    print 'Overall accuracy = {}\n'.format(acc)
    #print 'Slum accuracy = {}\n'.format(cm[0,0]/np.sum(cm[0,:]))
    print 'Confusion Matrix \n {}\n'.format(cm)
    print 'Classification Report \n {}\n'.format(cr)


    result = model.predict(testX)
    acc = accuracy_score(testY, result)
    cm = confusion_matrix(testY, result)
    cr = classification_report(testY,result, target_names=['Urban','Background'])
    print '{} CLassifier test results:'.format(clf_path)
    print 'Overall accuracy = {}\n'.format(acc)
    #print 'Slum accuracy = {}\n'.format(cm[0,0]/np.sum(cm[0,:]))
    print 'Confusion Matrix \n {}\n'.format(cm)
    print 'Classification Report \n {}\n'.format(cr)

    print "==================================================================="
    
    
print train_L2_Y.shape, test_L2_Y.shape

print train_L3_Y.shape, test_L3_Y.shape

# filter correctly classified training and test data
final_all_data_X = np.vstack((trainX, testX))
final_all_data_Y = np.hstack((trainY, testY))

print 'L1 data shape training: {}, L1 data shape testing: {}'.format(trainY.shape, testY.shape)
print 'L2 data shape training: {}, L2 data shape testing: {}'.format(train_L2_Y.shape, test_L2_Y.shape)
print 'L3 data shape training: {}, L3 data shape testing: {}'.format(train_L3_Y.shape, test_L3_Y.shape)
final_all_data_L2_Y = np.hstack((train_L2_Y, test_L2_Y))
final_all_data_L3_Y = np.hstack((train_L3_Y, test_L3_Y))

clf_path='xgboost'
model = pickle.load(open('../models/final-L1-{}-model.sav'.format(clf_path),'rb'))
result = model.predict(final_all_data_X)
    
print final_all_data_X.shape
result = model.predict(final_all_data_X)
final_all_out_X = final_all_data_X[result == final_all_data_Y,:] 
final_all_out_Y = final_all_data_Y[result == final_all_data_Y]
final_all_out_L2_Y = final_all_data_L2_Y[result == final_all_data_Y]
final_all_out_L3_Y = final_all_data_L3_Y[result == final_all_data_Y]


## Split the correctly classified instances into training and testing
train_proportion = 0.7

#extract only the data relevant to correctly classified instances for class informal
k = 0
curr_class_indices = np.asarray(np.where(final_all_out_Y == k)).ravel()
n_vals = curr_class_indices.shape[0]
np.random.shuffle(curr_class_indices)
n_train = int(train_proportion * n_vals)
print curr_class_indices.shape
curr_train_indices = curr_class_indices[0:n_train]
curr_test_indices = curr_class_indices[n_train:]
curr_train_X = final_all_out_X[curr_train_indices, :]
#curr_train_Y = final_all_out_Y_labels[curr_train_indices]
curr_test_X = final_all_out_X[curr_test_indices, :]
#curr_test_Y = final_all_out_Y_labels[curr_test_indices]
train_L2_Y = final_all_out_L2_Y[curr_train_indices]
test_L2_Y = final_all_out_L2_Y[curr_test_indices]
train_L3_Y = final_all_out_L3_Y[curr_train_indices]
test_L3_Y = final_all_out_L3_Y[curr_test_indices]


# Uncomment the next few lines to write these files to disk

'''
f = open('/home/kgadira/data/filtered-L1-L2-tr-X','w')
pickle.dump(curr_train_X,f)
f.close()

f = open('/home/kgadira/data/filtered-L1-L2-te-X','w')
pickle.dump(curr_test_X,f)
f.close()

f = open('/home/kgadira/data/filtered-L1-L2-tr-Y','w')
pickle.dump(train_L2_Y,f)
f.close()

f = open('/home/kgadira/data/filtered-L1-L2-te-Y','w')
pickle.dump(test_L2_Y,f)
f.close()

f = open('/home/kgadira/data/filtered-L1-L3-tr-Y','w')
pickle.dump(train_L3_Y,f)
f.close()

f = open('/home/kgadira/data/filtered-L1-L3-te-Y','w')
pickle.dump(test_L3_Y,f)
f.close()

'''
