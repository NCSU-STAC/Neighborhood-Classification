# Hierarchical classification at Level 1 {informal vs formal}
* Please note, you have to change paths to your data in each file. 
* File descriptions are given below. Please note, slumNet.py refers to the CNN.It is called by train and test functions. To run the train and test methods, change paths to your respective data, and run python trainL2.py (to train) and python testNet.py to run thesting. Descriptions of each method are given below: 
	* train*.py - to train the model for the current level of classification. Please note, you need to have folders named models and models/filtered-model-3-cl-final before running the network to store the  final model weights after training.
	* testNet.py - to verify the correctness of the model on the sample test set
	* slumNet.py - the neural net used for this purpose
	* predictHierarchy*.py - this performs the hierarchical filtration  by removing the misclassified instances from both training and test datasets. This data is then passed down to the next level (Hierarchical-Level 3)
* classifyImage.py - it classifies a 40 x 40 patch, jumps 40 pixels forward and clasifies next patch After finishing the row, jumps 40 rows down. THis movement is generated using the ground truth shape file, which is made up of a grid , with a point at the top left corner of each square in the grid. Given that creating a large raster for entire image is too memory consuming, you can specify the starting row and ending row as inputs to the code. To run this file, use ```python classifyImage.py starting_row_number ending_row_number```

