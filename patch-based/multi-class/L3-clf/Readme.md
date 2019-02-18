# Multi-class classification at Level 3 {single-story vs multi-story vs semi-permanent vs temporary vs formal vs background}
* Please note, you have to change paths to your data in each file. 
* File descriptions are given below. Please note, slumMultiNet.py refers to the CNN.It is called by train and test functions. To run the train and test methods, change paths to your respective data, and run python trainL3.py (to train) and python testNet.py to run thesting. Descriptions of each method are given below: 
	* train*.py - to train the model for the current level of classification. Please note, you need to have folders named models and models/model-6-cl-final1 before running the network to store the  final model weights after training.
	* testNet.py - to verify the correctness of the model on the sample test set
	* slumMultiNet.py - the neural net used for this purpose
