train*.py - to train the model for the current level of classification
testNet.py - to verify the correctness of the model on the sample test set
slumNet.py - the neural net used for this purpose
classifyImage.py - classify entire image, patch by patch with a stride = length of the patch
classifyImagePerPixel.py - classify entire image by taking a pixel, extract a patch around it, classifying the patch and assigning label to the pixel
