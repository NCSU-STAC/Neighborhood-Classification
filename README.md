# Neighborhood-Classification
Joint research project between NCSU, UNC Chapel Hill and Duke University to detect different types of low-income settlements from VHR imagery

## Necessary tools/softwares to be installed
1. python 2.7
2. Tensorflow
3. TFLearn
4. Scikit-learn
5. Scikit-image
6. GDAL
7. GDAL-python bindings
8. fiona - python package to read shape files
9. XGBoost - python package
10. ORFEO Toolbox (for feature generation)
11. QGIS (For visualizing results)
12. OpenCV 2.0
13. Jupyter (to process ipython notebooks)

## Necessary hardware
* For the CNN solution, it is expected that you have a tensorflow-capable GPU
* The pixel-based solutions do not require a GPU. 

## How to run?
* Instructions to run the files are all placed in the sub folders
* Unless explicitly stated, each python file can be run using ```python filename```
* Except for classifyImage python files, for which you ahve to give start and end row numbers as command line arguments (described in specific folders)


## Pipeline
1. Preprocessing:
* All images and any ancillary data should be in the same projection syystem. You can use GDAL to ensure all files in the same projection system.
* Mosaic all the separate tiles into a single image. Once again, you can use gdal for this purpose

2. Feature Generation
* You generate any features that are relevant to your research. You can use tools such as [ERDAS Imagine][https://www.hexagongeospatial.com/products/power-portfolio/erdas-imagine], [ENVI][https://www.harrisgeospatial.com/Software-Technology/ENVI], [QGIS][https://qgis.org/en/site/], ![Orfeo Toolbox][https://www.orfeo-toolbox.org/] to generate features. In this project, we generated Haralick texture features [9] using orfeo toolbox, NDBI [10] and edge density using python. Code for generating these features is available in the feature-generation folder.

3. Data collection
* Data was collected in QGIS. Save your shape file as a point shape file. Each point in the shapefile should have the following information: lat, long values in the same projection as your raster data (if you are using ESRI shapefile format, this will already be included in your shapefile), a column named 'Type' to indicate training or testing data, a column named 'Cl' to indicate class. Class values should start from 0. For eg, if you are building a dataset with {builtup, background}, 'Cl' value of building is 0 and 'Cl' value of background is 1. 
* Once this data is collected, use the scripts in the preprocessing folder to save the information as [pickle][https://docs.python.org/2/library/pickle.html] format to easily read and write them for all model training and evaluation purposes.
4. Training/Evaluation/Prediction:
* Once you have all the training, test data ready, generate the training models for each type of classification using the code in the respective folders.
	* For pixel-based(patch-based), use the pixel-based(patch-based) folder.  
	* We have two types of pixel-based (patch-based) classification. In the multi-class classification, you perform the following classifications:
		* multi-class/level1: built-up vs background
		* multi-class/level 2: informal vs formal vs background
		* multi-class/level 3: single-story vs multi-story vs semi-permeant vs temporary vs formal vs background
	* In hierarchical classification, you perform the following steps:
		1. hierarchical/level 1: built up vs background. Filter misclassified built up areas, identify corresponding informal and formal from this built-up subset, and proceed to level 2.
		2. hierarchical/level 2: informal vs formal. Filter misclassified informal instances, identify corresponding single-story, multi-story, semi-permanent and temporary informal instances, proceed to step 3.
		3. hierarchical/level 3: single-story vs multi-story vs semi-permanent vs temporary
* Once you are done building the models, evaluate the performance of your models by predicting on the test set.
* Predicting on entire image: Given that the size of the raster used in this project is very large, it is not possible to load the entire raster into memory to perform prediction on the entire image. You can either:
	*  Classify a region of interest: take a section of the image (by clipping a region of interest) and perform prediction on it. You can use the classifyImage* for this. However, you need to make sure that the dimensions of the image are given properly to this code. 
	* Classify entire image in sections: Use the same classifyImage* python files, give starting and ending rows of sections of the image. Save thes files to disk as numpy/pickle arrays, merge them together later using scripts from postprocessing folder. 
	* Parallel Solutions for pixel-based classification: A parallel MPI-based solution has been provided for pixel-based classification on entire image. This requires a multi-node HPC setup such ![ARC][http://moss.csc.ncsu.edu/~mueller/cluster/arc/].  This setup makes use of the existing classifyImage* python files, which classify the image between a starting row and ending row. The MPI setup repeatedly calls this classifyImage* repeatedly using different starting and ending rows, so that classification is done on different sections of the image in parallel. 
5. Change Detection: If you want to perform change detection to detect informal settlements from 2016 that were open land (or builtup area) in 2002, first classify the landsat image using pixel-based/landsat-clf methods (you need to generate haralick and other features for this data as well) and then use the corresponding file from change-detection folder.

## Citations
1.  D. Roy and D. Bernal, “An exploratory factor analysis model for slum severity index in Mexico City,” Tech. Rep., 2018.
2.  B. Pradhan, “Spatial Modeling and Assessment of Urban Form Analysis of Urban Growth: From Sprawl to Compact Using Geospatial Data,” Tech. Rep.
3.  R. R. Vatsavai, “Gaussian multiple instance learning approach for mapping the slums of the world using very high resolution imagery,” in Proceedings of the 19th ACM SIGKDD international conference on Knowledge discovery and data mining - KDD ’13, 2013.
4.  U. N. publication issued by the DeSA, “The Sustainable Development Goal Report 2017,” Tech. Rep.
5. O. Kit and M. Lüdeke, “Automated detection of slum area change in hyderabad, india using multitemporal satellite imagery,” ISPRS journal of photogrammetry and remote sensing, vol. 83, pp. 130–137, 2013.
6.  A. Sekertekin, A. Marangoz, and H. Akcin, “Pixel-based classification analysis of land use land cover using sentinel-2 and landsat-8 data,” Forest, vol. 80, no. 78.13, pp. 82–71, 2017.
7.  N. Mboga, C. Persello, J. Bergado, and A. Stein, “Detection of Informal Settlements from VHR Images Using Convolutional Neural Networks,” Remote Sensing, 2017.
8.  R. R. Vatsavai, “Scalable multi-instance learning approach for mapping the slums of the world,” in High Performance Computing, Networking, Storage and Analysis (SCC), 2012 SC Companion:. IEEE, 2012, pp. 833–837.
9. R. M. Haralick, K. Shanmugam et al., “Textural features for image classification,” IEEE Transactions on systems, man, and cybernetics, no. 6, pp. 610–621, 1973.
10. Y. Zha, J. Gao, and S. Ni, “International Journal of Remote Sensing Use of normalized difference built-up index in automatically mapping urban areas from TM imagery Use of normalized difference built-up index in automatically mapping urban areas from TM imagery,” 2010. [Online]. Available: http://www.tandfonline.com/action/journalInformation?journalCode=tres20<F12>
11. J. Canny, “A Computational Approach to Edge Detection,” IEEE Transactions on Pattern Analysis and Machine Intelligence, 1986. 
12. N. Jean, M. Burke, M. Xie, W. M. Davis, D. B. Lobell, and S. Ermon, “Combining satellite imagery and machine learning to predict poverty,” Tech. Rep. [Online]. Available: http://science.sciencemag.org/
13. Pedregosa, Fabian, Gaël Varoquaux, Alexandre Gramfort, Vincent Michel, Bertrand Thirion, Olivier Grisel, Mathieu Blondel et al. "Scikit-learn: Machine learning in Python." Journal of machine learning research 12, no. Oct (2011): 2825-2830.
14. Inglada, Jordi, and Emmanuel Christophe. "The Orfeo Toolbox remote sensing image processing software." In Geoscience and Remote Sensing Symposium, 2009 IEEE International, IGARSS 2009, vol. 4, pp. IV-733. IEEE, 2009.
15. https://www.machinelearningmastery.com
16. Digital Globe 
17. USGS Earth Explorer
18. Fiona (python package for processing shape files)
19. TFLearn: Damien, Aymeric. "Tflearn." (2016).
