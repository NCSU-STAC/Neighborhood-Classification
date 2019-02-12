'''
Picks a patch surrounding each pixel, classifies the patch and assigns the label to the pixel
Since the entire image is too large, we take a subset of rows, classify them and save them as numpy arrays to be combined later
@author: Krishna Karthik Gadiraju/kkgadiraju
'''

from osgeo import gdal
import ogr, osr
import numpy as np
import fiona, sys
import slumnet as net
from sklearn.feature_extraction.image import extract_patches_2d
import time
import logging
import socket

host = socket.gethostname()
logging.basicConfig( filename = 'Type2-ImageClf-{}.log'.format(host),format='%(asctime)s %(levelname)-8s %(message)s', level=logging.DEBUG, datefmt='%Y-%m-%d %H:%M:%S')
# predict for entire image, patch by patch
driver = gdal.GetDriverByName('GTiff')
rasterFileName1 = "/pvfs2/kgadira@oss-storage-0-108/pvfs2/kgadira/data/PS_mosaic_415.tif"
dataset1 = gdal.Open(rasterFileName1)
rasterFileName2 = "/pvfs2/kgadira@oss-storage-0-108/pvfs2/kgadira/data/PS_mosaic_415_NDBI.tif"
dataset2 = gdal.Open(rasterFileName2)
rasterFileName3 = "/pvfs2/kgadira@oss-storage-0-108/pvfs2/kgadira/data/PAN_mosaic_415-simple-50.tif"
dataset3 = gdal.Open(rasterFileName3)
rasterFileName4 = "/pvfs2/kgadira@oss-storage-0-108/pvfs2/kgadira/data/PAN_mosaic_415_edgeDensity.tif"
dataset4 = gdal.Open(rasterFileName4)
#vector = fiona.open('/scratch/slums/bl-slums/gt/final-data/gt10e/gt10e.shp')
imggeotrans = dataset1.GetGeoTransform()
cols = dataset1.RasterXSize
rows = dataset1.RasterYSize
transform = dataset1.GetGeoTransform()
xOrigin = transform[0]
yOrigin = transform[3]
pixelWidth = transform[1]
pixelHeight = -transform[5]
bands1=[]
bands2=[]
bands3=[]
bands4=[]

for i in range(8):
    bands1.append(dataset1.GetRasterBand(i+1)) 
bands2.append(dataset2.GetRasterBand(1)) 

for i in range(8):
    bands3.append(dataset3.GetRasterBand(i+1)) 
bands4.append(dataset4.GetRasterBand(1)) 

#print rows, cols
actual_class = []
predicted_class = []

START = int(sys.argv[1])
END = int(sys.argv[2])

logging.info('START INDEX = {}, END INDEX = {}'.format(START, END))

# Get the model
path_to_model = '../models/model-3-cl-final/final-model.tflearn'
model = net.model
model.load(path_to_model)
i = 0
start_t = time.time()
result_raster = np.zeros((END-START,cols), dtype = int)-1
noneData = False
BATCH_SIZE = 3000
for curr_row in range(START, END):
    #print curr_row
    curr_images = np.zeros((18, 40, cols)) + np.Inf
    for k in range(8):
        data = bands1[k].ReadAsArray(0,curr_row-20, cols, 40)
        print data.shape
        curr_images[k, :, :] = data
    curr_images[8, :, :] = bands2[0].ReadAsArray(0,curr_row-20,cols, 40)
    for k in range(8):
        curr_images[9+k, :, :] = bands3[k].ReadAsArray(0,curr_row-20,cols, 40)

    curr_images[17, :, :] = bands4[0].ReadAsArray(0,curr_row-20,cols, 40)
    curr_images = curr_images.T
    curr_images = np.nan_to_num(curr_images)
    #print curr_images.shape
    curr_images_sliding_windows =  extract_patches_2d(curr_images, (40, 40))
    # print curr_images_sliding_windows.shape
    n_windows = curr_images_sliding_windows.shape[0]
    for i in range(0, n_windows + 1, BATCH_SIZE):
        curr_batch = curr_images_sliding_windows[i: (i + BATCH_SIZE),:,:, :]
        result = model.predict(curr_batch) # Predict
        #print result
        prediction = np.argmax(result, axis = 1)
        #logging.info('Predicion shape = {}, result raster subset shape = {}'.format(prediction.shape, result_raster[curr_row, (i+19) : min(i+ 19 + BATCH_SIZE, n_windows)].shape))
        result_raster[curr_row-START, (i+19) : min(i+ 19 + BATCH_SIZE, n_windows+19)] = prediction
        #print result, prediction
        #print prediction.shape
    if curr_row % 10 == 0:
        logging.info('Finished {} rows'.format(curr_row))

end_t = time.time()

print 'Total exec time = {}'.format(end_t-start_t)
    

f = '/pvfs2/kgadira@oss-storage-0-108/pvfs2/kgadira/data/CNN-clf-T2-prediction-{}-{}'.format(START,END)
np.save(f, result_raster)

#write_image(cols, rows, 1,imggeotrans, result_raster, '/scratch/slums/bl-slums/clf-img/AllImg_L1_MASK.tif')
