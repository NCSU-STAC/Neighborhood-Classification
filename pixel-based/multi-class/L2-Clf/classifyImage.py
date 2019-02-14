from osgeo import gdal
import ogr
import numpy as np
import time, sys, os
import pickle



def write_image(imgrows, imgcols, imgbands, imggeotrans,edges, opath):
    driver = gdal.GetDriverByName('GTiff')
    outRaster = driver.Create(opath, imgrows, imgcols, imgbands, gdal.GDT_Float32)
    outRaster.SetGeoTransform(imggeotrans)
    outband = outRaster.GetRasterBand(1)
    outband.WriteArray(edges)
    outRasterSRS = osr.SpatialReference()
    outRasterSRS.ImportFromEPSG(32643)
    outRaster.SetProjection(outRasterSRS.ExportToWkt())
    outband.FlushCache()



clf_names = ['rf','adaboost','mlp']

driver = gdal.GetDriverByName('GTiff')
rasterFileName1 = "/pvfs2/kgadira@oss-storage-0-108/pvfs2/kgadira/data/PS_mosaic_415.tif"
dataset1 = gdal.Open(rasterFileName1)
rasterFileName2 = "/pvfs2/kgadira@oss-storage-0-108/pvfs2/kgadira/data/PS_mosaic_415_NDBI.tif"
dataset2 = gdal.Open(rasterFileName2)
rasterFileName3 = "/pvfs2/kgadira@oss-storage-0-108/pvfs2/kgadira/data/PAN_mosaic_415-simple-50.tif"
dataset3 = gdal.Open(rasterFileName3)
rasterFileName4 = "/pvfs2/kgadira@oss-storage-0-108/pvfs2/kgadira/data/PAN_mosaic_415_edgeDensity.tif"
dataset4 = gdal.Open(rasterFileName4)


imggeotrans = dataset1.GetGeoTransform()


bands1=[]
bands2=[]
bands3=[]
bands4=[]
data_all_bands = []
cols = dataset1.RasterXSize
rows = dataset1.RasterYSize

transform = dataset1.GetGeoTransform()
xOrigin = transform[0]
yOrigin = transform[3]
pixelWidth = transform[1]
pixelHeight = -transform[5]

print xOrigin, yOrigin, pixelWidth, pixelHeight

for i in range(8):
    bands1.append(dataset1.GetRasterBand(i+1)) 
    # data_all_bands.append(band.ReadAsArray(0,0,cols,rows).astype(np.float))
bands2.append(dataset2.GetRasterBand(1)) 
for i in range(8):
    bands3.append(dataset3.GetRasterBand(i+1)) 
bands4.append(dataset4.GetRasterBand(1)) 

print rows, cols
SPLIT_SIZE = 1000
splits = []
for i in range(0, cols, SPLIT_SIZE):
    splits.append(i)
splits.append(cols-1)
# print splits
read_time = 0
clf_time = 0

MININDEX = int(sys.argv[1])
MAXINDEX = int(sys.argv[2])


for clf_path in clf_names: 
    # check if you've already classfied this part of the image - if yes, return
    f = '/home/kgadira/results/{}-clf-L2-{}-{}.bin.npy'.format(clf_path, MININDEX, MAXINDEX)
    if os.path.isfile(f):
        print 'Processing already completed for{} for the image'.format(clf_path)
        continue
    model = pickle.load(open('../models/final-L2-{}-model.sav'.format(clf_path),'rb'))
    start_t = time.time()
    result_raster = np.zeros((MAXINDEX-MININDEX,cols), dtype=np.int8)-1
    noneData = False
    for i in range(MININDEX, MAXINDEX):
        r_start = time.time()
        curr_row = np.zeros((cols,18))-1
        for k in range(8):
            data = bands1[k].ReadAsArray(0,i,cols,1)
            data =data.astype(np.float)
            #print data.shape
            curr_row[:,k] = data 
        for k in range(8):
            data = bands3[k].ReadAsArray(0,i,cols,1)
            #print data.shape
            data = data.astype(np.float)
            curr_row[:,k+8] = data 
        data = bands4[0].ReadAsArray(0,i,cols,1)
        curr_row[:,17] = data
	r_end = time.time()
	read_time += r_end - r_start
	#print 'Finished reading line'
        for k in range(len(splits)-1):
            curr_Y = curr_row[splits[k]:splits[k+1],:]
            curr_Y = np.nan_to_num(curr_Y)
	    result = model.predict(curr_Y)    
            result_raster[i-MININDEX,splits[k]:splits[k+1]] = result
            #print 'Finished {} pixels in {} row'.format(splits[k],i)
        clf_time += time.time() - r_end   
     	if i-MININDEX % 10 == 0 and i-MININDEX > 0:
            print 'Finished {} lines, clf time = {} '.format(i, clf_time)
        print 'Reading time = {}'.format(read_time)
        clf_time = 0
	read_time = 0
    end_t = time.time()

    print 'Total exec time = {}'.format(end_t-start_t)
    result_raster = result_raster.astype(np.int8)
    f = '/home/kgadira/results/{}-clf-L2-{}-{}.bin'.format(clf_path, MININDEX, MAXINDEX)
    np.save(f,result_raster)
    #io.write_image(cols, MAXINDEX-MININDEX, 1,imggeotrans, result_raster, '/home/kgadira/results/{}-clf-3Class-{}-{}.tif'.format(clf-path, MININDEX, MAXINDEX)) 
    del result_raster
    del result

