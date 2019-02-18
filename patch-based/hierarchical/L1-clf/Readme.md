# Hierarchical classification at Level 1 {built-up vs background}
* Please note, you have to change paths to your data in each file. 
* File descriptions are given below. Please note, buildingNet.py refers to the CNN.It is called by train and test functions. To run the train and test methods, change paths to your respective data, and run python trainL3.py (to train) and python testNet.py to run thesting. Descriptions of each method are given below: 
	* train*.py - to train the model for the current level of classification. Please note, you need to have folders named models and models/filtered-model-2-cl-final before running the network to store the  final model weights after training.
	* testNet.py - to verify the correctness of the model on the sample test set
	* buildingNet.py - the neural net used for this purpose
	* predictHierarchy*.py - this performs the hierarchical filtration  by removing the misclassified instances from both training and test datasets. This data is then passed down to the next level (Hierarchical-Level 2)
