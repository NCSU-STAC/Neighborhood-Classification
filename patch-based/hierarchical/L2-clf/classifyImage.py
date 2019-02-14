from osgeo import gdal
import ogr, osr
import numpy as np
import fiona, sys
import slumnet as net

import time

# predict for entire image, patch by patch
driver = gdal.GetDriverByName('GTiff')
rasterFileName1 = "/home/kgadira/data/PS_mosaic_415.tif"
dataset1 = gdal.Open(rasterFileName1)
rasterFileName2 = "/home/kgadira/data/PS_mosaic_415_NDBI.tif"
dataset2 = gdal.Open(rasterFileName2)
rasterFileName3 = "/home/kgadira/data/PAN_mosaic_415-simple-50.tif"
dataset3 = gdal.Open(rasterFileName3)
rasterFileName4 = "/home/kgadira/data/PAN_mosaic_415_edgeDensity.tif"
dataset4 = gdal.Open(rasterFileName4)
vector = fiona.open('/home/kgadira/data/gt10e/gt10e.shp')
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

print rows, cols
actual_class = []
predicted_class = []

START = int(sys.argv[1])
END = int(sys.argv[2])
# Get the model
path_to_model = '../models/filtered-model-3-cl-final/final-model.tflearn'
model = net.model
model.load(path_to_model)
i = 0
start_t = time.time()
result_raster = np.zeros((END-START,cols), dtype = int)-1
noneData = False
for feat in vector:
    #if i%1000==0:
    #    print 'Finished {} patches'.format(i)
    #curr_class = feat['properties']['Cl']
    curr_coords = feat['geometry']['coordinates']
    curr_img = np.zeros((18,40,40))-1
    curr_img = curr_img.astype(float)
    col = int((curr_coords[0] - xOrigin) / pixelWidth)
    row = int((yOrigin - curr_coords[1] ) / pixelHeight)
    if row < START:
        continue
    if row>END:
        break
    if col+40 > cols:
        continue
    for k in range(8):
        data = bands1[k].ReadAsArray(col,row,40,40)
        if data is None:
            #actual_class.append(curr_class)
            #predicted_class.append(-1)
            noneData = True
            break
        data =data.astype(np.float)
        curr_img[k,:,:] = data
    if noneData:
        noneData = False
        continue
    data = bands2[0].ReadAsArray(col,row,40,40)
    if data is None:
        #actual_class.append(curr_class)
        #predicted_class.append(-1)
        noneData = True
        break
    data =data.astype(np.float)
    curr_img[8,:,:] = data

    if noneData:
        noneData = False
        continue
    for k in range(8):
        data = bands3[k].ReadAsArray(col,row,40,40)
        if data is None:
            #actual_class.append(curr_class)
            #predicted_class.append(-1)
            noneData = True
            break
        data =data.astype(np.float)
        curr_img[k+9,:,:] = data
    if noneData:
        noneData = False
        continue
    data = bands4[0].ReadAsArray(col,row,40,40)
    curr_img[17,:,:] = data
    if data is None:
        #actual_class.append(curr_class)
        #predicted_class.append(-1)
        noneData = True
        break
    
    if i == 0:
        print curr_img.shape
    curr_img = curr_img.T
    if i == 0:
        print curr_img.shape
    curr_img = curr_img.reshape([-1, 40, 40, 18])
    if i == 0:    
        print curr_img.shape
    result = model.predict(curr_img)[0] # Predict
    #print result
    prediction = np.argmax(result)

    #print prediction
    predicted_class.append(prediction)
    result_raster[row-START:row-START+40,col:col+40] = prediction
    i+=1
    if i % 5000 == 0 and i > 0:
        out_str = '\r Row number = {} | Patch ID# = {} | probabilities = {} | class = {}\n'.format(row,feat['properties']['ID'],result ,prediction)
        print '{}'.format(out_str)
        sys.stdout.flush()
        
end_t = time.time()

print 'Total exec time = {}'.format(end_t-start_t)
    

f = '/home/kgadira/results/CNN-filtered-clf-L2-{}-{}.bin'.format(START, END)
np.save(f, result_raster)
#write_image(cols, END-START, 1,imggeotrans, result_raster, '/home/kgadira/results/AllImg_L1_MASK-{}-{}.tif'.format(START,END))
