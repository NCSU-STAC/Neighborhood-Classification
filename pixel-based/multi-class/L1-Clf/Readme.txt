### L1 classification per pixel
ipython notebook does the grid search and saves models
Model generation is being done separately in another file (generate_training_models*.py) since the machine we were using for hyperparameter search and the machine we were using for inference (prediction across entire image) do not have the same versions of necessary software.
classifyImage.py - to classify section of the entire image - takes a range of rows, (starting row, ending row) - will classify this section
we use predict.c which uses MPI on NCSU's ARC cluster to generate results in distributed manner
