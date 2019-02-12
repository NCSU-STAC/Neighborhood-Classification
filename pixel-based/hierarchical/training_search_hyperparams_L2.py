'''
Generate optimal hyperparameters using filtered training data from L1 classification
@author: Krishna Karthik Gadiraju/kkgadiraju
Source: Turorials and API from scikit-learn website

'''


import numpy as np
import xgboost
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier

# Set seed
np.random.seed(100)

# Read files 
trainX = np.load('/scratch/slums/bl-slums/gt/filtered-L1-L2-tr-X')
trainY = np.load('/scratch/slums/bl-slums/gt/filtered-L1-L2-tr-Y')
testX = np.load('/scratch/slums/bl-slums/gt/filtered-L1-L2-te-X')
testY = np.load('/scratch/slums/bl-slums/gt/filtered-L1-L2-te-Y')

trainY = trainY.ravel()
testY = testY.ravel()
print trainX.shape, trainY.shape, testX.shape, testY.shape

# Utility function to report best scores
# Source stackoverflow and scikitlearn-website
def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))


#XGB CLassifier
model = xgboost.XGBClassifier(nthread=16 ,objective='binary:logistic')
learning_rate = [0.2, 0.3, 0.5, 0.7, 0.9]
n_estimators = [1000,5000,6000, 7000, 10000]
param_grid = dict(learning_rate=learning_rate, n_estimators = n_estimators)
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=7)
grid_search = GridSearchCV(model, param_grid, scoring="neg_log_loss", n_jobs=-1, cv=kfold)
grid_result = grid_search.fit(trainX, trainY)
report(grid_result.cv_results_)



# Random Forest Classifier
rf = RandomForestClassifier(n_estimators = 500)
param_grid = {"max_depth": [3, None],
              "min_samples_split": [2, 3],
              "min_samples_leaf": [1, 3],
              "n_estimators": [500, 1000],
              "bootstrap": [True, False],
              "criterion": ["gini", "entropy"]}
grid_search = GridSearchCV(rf, param_grid = param_grid, scoring="neg_log_loss", n_jobs=-1, cv=kfold)
grid_result = grid_search.fit(trainX, trainY)
report(grid_result.cv_results_)

# KNN Classifier
cv = StratifiedKFold(n_splits = 10, random_state =7 )
knn = KNeighborsClassifier()
n_neighbors = list(np.arange(3,11,1))
#print n_neighs
params_grid = dict(n_neighbors= n_neighbors)
knn_grid_search = GridSearchCV(estimator = knn, n_jobs = -1, param_grid = params_grid)
knn_grid_result = knn_grid_search.fit(trainX, trainY)
report(knn_grid_result.cv_results_)

# MLP Classifier
mlp = MLPClassifier()
params_grid={
'learning_rate': ["constant", "invscaling", "adaptive"],
'hidden_layer_sizes': [(100,100,100,100), (100,100,100,100,100), (100,100,100,100,100,100,100,100)],
'alpha': [0.0001, 0.00001, 0.01],
'activation': ["logistic", "relu", "tanh"]
}

mlp_grid_search = GridSearchCV(estimator=mlp,param_grid=params_grid,n_jobs=-1,cv=kfold)
mlp_grid_result = mlp_grid_search.fit(trainX, trainY)
report(mlp_grid_result.cv_results_)

# Adaboost classifier
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=7)
adb = AdaBoostClassifier()
params_grid = dict(n_estimators=[50,100,500,1000,5000], learning_rate=[0.01, 0.007, 0.0001, 0.1, 0.0007])
adb_grid_search = GridSearchCV(estimator=adb, param_grid = params_grid, cv=kfold)
adb_search_result = adb_grid_search.fit(trainX, trainY)
report(adb_search_result.cv_results_)
