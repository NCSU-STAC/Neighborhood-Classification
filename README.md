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

## How to run?
*Instructions to run the files are all placed in the sub folders
*Unless explicitly stated, each python file can be run using ```python filename```
*Except for classifyImage python files, for which you ahve to give start and end row numbers as command line arguments (described in specific folders)

## Pipeline
1. First generate features either using the code in feature-generation folder, or using any tool/software such as ARCGIS/ENVI. You need NDBI, Haralick Texture Features, Edge Density as features
2. Generate training data as pickle files, so that you can easily read and write them for all files using respective codes from preprocessing folders
3. Generate the training models for each type of pixel/patch based/multi-class/hierarchical classification. For pixel based methods, you do a grid search for to identify the best parameters for a classifer. For patch absed, you use a CNN
4. Predict on the test set to verify how well your model is behaving
5. To predict on a larger section of the image, use the classifyImage.. python files
6. Use the postprocessing scripts to merge all the sub section of results generated for the entire big image.
7. If you want to perform change detection to detect informal settlements from 2016 that were open land (or builtup area) in 2002, first classify the landsat image using pixel-based/landsat-clf methods (you need to generate haralick and other features for this data as well) and then use the corresponding file from change-detection folder.

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
16. Digital GLobe 
17. USGS Earth Explorer
18. Fiona (python package for processing shape files)
