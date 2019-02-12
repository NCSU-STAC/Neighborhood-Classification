'''
NDBI Calculation
NDBI = (SWIR - NIR)/(SWIR + NIR)
In VHR, band 8 (index 7) is NIR2 - assming it as SWIR, band 7 (index 6) is NIR 1 - assuming it is NIR
@author: Krishna Karthik Gadiraju/kkgadiraju
'''

import cv2
import struct
import random
import gdal, ogr, osr
from gdalconst import *
import numpy as np
from sklearn.preprocessing import normalize
from scipy import stats
from sklearn.externals.joblib.parallel import Parallel, delayed
import multiprocessing
import time

NUM_REGIONS=5

def _generate_regions_of_interest_indices(imgrows,imgcols, num_regions):
        print('generating ROIs for rows = %d, cols = %d ' %(imgrows,imgcols)) 
	split_row = int(imgrows/num_regions)
        split_col = int(imgcols/num_regions)
        region_rows = [min(i*split_row, imgrows) for i in range(num_regions+1)]
        region_cols = [min(i*split_col, imgcols) for i in range(num_regions+1)]
	return([region_rows, region_cols])


def read_image(ipath):	
	i1 = gdal.Open(ipath , GA_ReadOnly )
	imgrows = i1.RasterXSize #Get raster information
	imgcols = i1.RasterYSize
	imgbands = i1.RasterCount
	imggeotrans = i1.GetGeoTransform()
	return([i1, imgrows, imgcols, imgbands, imggeotrans])

def write_image(imgrows, imgcols, imgbands, imggeotrans,edges, opath):
	driver = gdal.GetDriverByName('GTiff')
    	outRaster = driver.Create(opath, imgrows, imgcols, 1, gdal.GDT_Float32)
    	outRaster.SetGeoTransform(imggeotrans)
    	outband = outRaster.GetRasterBand(1)
    	outband.WriteArray(edges)
    	outRasterSRS = osr.SpatialReference()
    	outRasterSRS.ImportFromEPSG(32643)
    	outRaster.SetProjection(outRasterSRS.ExportToWkt())
    	outband.FlushCache()
	
#calculate NDVI values (NIR-R)/(NIR + R)
def generate_NDVI():	
	ipath = '/scratch/slums/bl-slums/raw-img/MUL_mosaic_415.tif'
	[i1, imgrows, imgcols, imgbands, imggeotrans] = read_image(ipath)
	print(imgrows, imgcols)
	[roi_rows, roi_cols] = _generate_regions_of_interest_indices(imgrows, imgcols, NUM_REGIONS)
	print(roi_rows, roi_cols)
        if(roi_rows[len(roi_rows)-1] < imgrows):
           roi_rows.append(imgrows)
        if(roi_cols[len(roi_cols)-1] < imgcols):
           roi_cols.append(imgcols)
        ndvi = np.zeros((imgcols, imgrows), dtype = np.float)
	imgMin = np.inf
	imgMax = -np.inf
        nir = i1.GetRasterBand(6)		
        swir = i1.GetRasterBand(7)

	for i in range(len(roi_rows)-1):
                #print b.XSize, b.YSize	
		curr_min_row = roi_rows[i]
		curr_max_row = roi_rows[i+1]-1
                for j in range(len(roi_cols)-1):
                	curr_min_col = roi_cols[j]
                	curr_max_col = roi_cols[j+1]-1
			curr_row_size = curr_max_row - curr_min_row
                	curr_col_size = curr_max_col - curr_min_col
			print 'Current row offset: %d, Current col offset:  %d, Currrent xsize : %d, Current y size = %d' %(curr_min_row, curr_min_col, curr_row_size, curr_col_size)
			ir = nir.ReadAsArray(roi_rows[i],roi_cols[j], curr_row_size ,curr_col_size).astype(np.float)
			sw = swir.ReadAsArray(roi_rows[i],roi_cols[j], curr_row_size ,curr_col_size).astype(np.float)
			sw = np.nan_to_num(sw)
                        ir = np.nan_to_num(ir)
                        start = time.time()
			ndvi[curr_min_col:curr_max_col, curr_min_row:curr_max_row] = (sw - ir)/(sw + ir)	
			print('Completed NDBI calculation for curr starting row = %d, curr starting col = %d, Number of pixels = %d, time = %s seconds' %(curr_min_row, curr_min_col, curr_row_size*curr_col_size, time.time() - start))
	return([imgrows, imgcols, imgbands, imggeotrans, ndvi])



if __name__ == "__main__":
	print 'Starting...'
	[imgrows, imgcols, imgbands, imggeotrans, edges] = generate_NDVI()
	opath = '/scratch/slums/bl-slums/raw-img/MUL_mosaic_415_NDBI.tif'
	write_image(imgrows, imgcols,imgbands, imggeotrans,edges, opath)
