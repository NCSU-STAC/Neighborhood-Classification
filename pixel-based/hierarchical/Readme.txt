## Hierarchical classification

L1 classification - Urban vs Other
L2 Classification - Informal vs Formal
L3 Classification - 4 different categories of Informal

Each file can be run by using the command: python <filename>

First run L1 classification, filter Urban class data - then filter the correctly classified urban instances, and train L2 classification - and now filter correctly classify informal instances and perform L3 classificaiton

Files:
training_search_hyperparameters*.py - To perform grid search to find the optimal parameters for each classifier
generate_training_models*.py - To train the data from the optimal parameters from training_search_hyperparameters*.py file and store models in ../models folder
predictHierarchy*.py - Load model created by generate_training_models*.py, run classifier, note training and test accuracy and then filter correctly classified instances into a separate matrix to be used by next level classification
classifyImage.py - classify entire Bangalore raster
