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
trainX = np.load('/home/kgadira/data/filtered-L2-L3-tr-X')
trainY = np.load('/home/kgadira/data/filtered-L2-L3-tr-Y')
testX = np.load('/home/kgadira/data/filtered-L2-L3-te-X')
testY = np.load('/home/kgadira/data/filtered-L2-L3-te-Y')



trainY = trainY.ravel()
testY = testY.ravel()

print trainY.shape, testY.shape
models_list = ['xgboost','rf','dt','gnb','knn','mlp','adaboost']
#trainX = np.nan_to_num(trainX)
#testX = np.nan_to_num(testX)
for clf_path in models_list:

    print "==================================================================="
    
    model = pickle.load(open('../models/filtered-L3-{}-model.sav'.format(clf_path),'rb'))
    result = model.predict(trainX)
    acc = accuracy_score(trainY, result)
    cm = confusion_matrix(trainY, result)
    cr = classification_report(trainY,result, target_names=['S1','S2','S3','S4'])
    print '{} Classifier training results:'.format(clf_path)
    print 'Overall accuracy = {}\n'.format(acc)
    #print 'Slum accuracy = {}\n'.format(cm[0,0]/np.sum(cm[0,:]))
    print 'Confusion Matrix \n {}\n'.format(cm)
    print 'Classification Report \n {}\n'.format(cr)


    result = model.predict(testX)
    acc = accuracy_score(testY, result)
    cm = confusion_matrix(testY, result)
    cr = classification_report(testY,result, target_names=['S1','S2','S3','S4'])
    print '{} CLassifier test results:'.format(clf_path)
    print 'Overall accuracy = {}\n'.format(acc)
    #print 'Slum accuracy = {}\n'.format(cm[0,0]/np.sum(cm[0,:]))
    print 'Confusion Matrix \n {}\n'.format(cm)
    print 'Classification Report \n {}\n'.format(cr)

    print "==================================================================="
    
    
