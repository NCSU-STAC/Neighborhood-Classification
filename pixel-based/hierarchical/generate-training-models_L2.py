'''
Use this file to generate training models on NCSU's ARC cluster
Note that grid search for finding the optimal hyperparameter configurations is done in another file (training_search_hyperparams*.py)
@author: Krishna Karthik Gadiraju/kkgadiraju
Source: Tutorials and API from scikit-learn website
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
trainX = np.load('/pvfs2/kgadira@oss-storage-0-108/pvfs2/kgadira/data/filtered-L1-L2-tr-X')
trainY = np.load('/pvfs2/kgadira@oss-storage-0-108/pvfs2/kgadira/data/filtered-L1-L2-tr-Y')
testX = np.load('/pvfs2/kgadira@oss-storage-0-108/pvfs2/kgadira/data/filtered-L1-L2-te-X')
testY = np.load('/pvfs2/kgadira@oss-storage-0-108/pvfs2/kgadira/data/filtered-L1-L2-te-Y')

trainY = trainY.ravel()
testY = testY.ravel()
print trainX.shape, trainY.shape, testX.shape, testY.shape

print "==================================================================="


#trainX = np.nan_to_num(trainX)
#testX = np.nan_to_num(testX)

xgb = xgboost.XGBClassifier(max_depth=1000, n_estimators=1000, nthread=16, objective='binary:logistic', learning_rate = 0.2 )

xgb.fit(trainX,trainY)
result = xgb.predict(testX)
acc = accuracy_score(testY, result)
cm = confusion_matrix(testY, result)
cr = classification_report(testY,result, target_names=['Informal','Formal'])
print 'XGB CLassifier results:'
print 'Overall accuracy = {}\n'.format(acc)
#print 'Slum accuracy = {}\n'.format(cm[0,0]/np.sum(cm[0,:]))
print 'Confusion Matrix \n {}\n'.format(cm)
print 'Classification Report \n {}\n'.format(cr)

fname = './models/filtered-L2-xgboost-model.sav'
pickle.dump(xgb, open(fname, 'wb'))

print "==================================================================="


rf = RandomForestClassifier(criterion='entropy',n_estimators=1000,min_samples_split = 3, max_depth = None, n_jobs=-1, bootstrap = True, min_samples_leaf = 1)
rf.fit(trainX,trainY)
result = rf.predict(testX)
acc = accuracy_score(testY, result)
cm = confusion_matrix(testY, result)
cr = classification_report(testY,result, target_names=['Informal','Formal'])
print 'RF CLassifier results:'
print 'Overall accuracy = {}\n'.format(acc)
#print 'Slum accuracy = {}\n'.format(cm[0,0]/np.sum(cm[0,:]))
print 'Confusion Madtrix \n {}\n'.format(cm)
print 'Classification Report \n {}\n'.format(cr)

fname = './models/filtered-L2-rf-model.sav'
pickle.dump(rf, open(fname, 'wb'))

print "==================================================================="


nb = GaussianNB()
nb.fit(trainX,trainY)
result = nb.predict(testX)
acc = accuracy_score(testY, result)
cm = confusion_matrix(testY, result)
cr = classification_report(testY,result, target_names=['Informal','Formal'])
print 'GNB CLassifier results:'
print 'Overall accuracy = {}\n'.format(acc)
print 'Confusion Matrix \n {}\n'.format(cm)
print 'Classification Report \n {}\n'.format(cr)

fname = './models/filtered-L2-gnb-model.sav'
pickle.dump(nb, open(fname, 'wb'))

print "==================================================================="


dt = DecisionTreeClassifier(random_state=100)
dt.fit(trainX,trainY)
result = dt.predict(testX)
acc = accuracy_score(testY, result)
cm = confusion_matrix(testY, result)
cr = classification_report(testY,result, target_names=['Informal','Formal'])
print 'DT CLassifier results:'
print 'Overall accuracy = {}\n'.format(acc)
#print 'Slum accuracy = {}\n'.format(cm[0,0]/np.sum(cm[0,:]))
print 'Confusion Matrix \n {}\n'.format(cm)
print 'Classification Report \n {}\n'.format(cr)

fname = './models/filtered-L2-dt-model.sav'
pickle.dump(dt, open(fname, 'wb'))

print "==================================================================="


knn = KNeighborsClassifier(9)
knn.fit(trainX, trainY)
result= knn.predict(testX)
acc = accuracy_score(testY, result)
cm = confusion_matrix(testY, result)
cr = classification_report(testY,result, target_names=['Informal','Formal'])
print 'KNN CLassifier results:'
print 'Overall accuracy = {}\n'.format(acc)
#print 'Slum accuracy = {}\n'.format(cm[0,0]/np.sum(cm[0,:]))
print 'Confusion Matrix \n {}\n'.format(cm)
print 'Classification Report \n {}\n'.format(cr)

fname = './models/filtered-L2-knn-model.sav'
pickle.dump(knn, open(fname, 'wb'))

print "==================================================================="


mlp = MLPClassifier(hidden_layer_sizes = (100,100,100,100), activation = 'tanh', learning_rate = 'constant', alpha = 0.0001)
mlp.fit(trainX, trainY)
result = mlp.predict(testX)
acc = accuracy_score(testY, result)
cm = confusion_matrix(testY, result)
cr = classification_report(testY,result, target_names=['Informal','Formal'])
print 'MLP CLassifier results:'
print 'Overall accuracy = {}\n'.format(acc)
#print 'Slum accuracy = {}\n'.format(cm[0,0]/np.sum(cm[0,:]))
print 'Confusion Matrix \n {}\n'.format(cm)
print 'Classification Report \n {}\n'.format(cr)

fname = './models/filtered-L2-mlp-model.sav'
pickle.dump(mlp, open(fname, 'wb'))

print "==================================================================="


adb = AdaBoostClassifier(n_estimators = 1000,learning_rate = 0.1)
adb.fit(trainX, trainY)
result = adb.predict(testX)
acc = accuracy_score(testY, result)
cm = confusion_matrix(testY, result)
cr = classification_report(testY,result, target_names=['Informal','Formal'])
print 'ADB CLassifier results:'
print 'Overall accuracy = {}\n'.format(acc)
#print 'Slum accuracy = {}\n'.format(cm[0,0]/np.sum(cm[0,:]))
print 'Confusion Matrix \n {}\n'.format(cm)
print 'Classification Report \n {}\n'.format(cr)

fname = './models/filtered-L2-adaboost-model.sav'
pickle.dump(adb, open(fname, 'wb'))

print "==================================================================="
