### Jupyter notebooks for landsat classification
* Preprocessing* ipython notebook: Given your ground truth shape file, this code reads the pansharpened 8-band LANDSAT raster (15 m resolution), haralick texture features, ndbi raster and edge density raster and saves the relevant information as a pickle file for easy read/write.
* Pixel-based* ipython notebook: performs training using grid search for several classifiers, generates models and classifies entire landsat image. 
