'''
Edge Density calculation
Detect edges using Canny Edge Detection
@author: Krishna Karthik Gadiraju/kkgadiraju
Source: http://docs.opencv.org/3.1.0/da/d22/tutorial_py_canny.html
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

NUM_REGIONS=150

def _generate_regions_of_interest_indices(imgrows,imgcols, num_regions):
        print('generating ROIs for rows = %d, cols = %d ' %(imgrows,imgcols)) 
	split_row = int(imgrows/num_regions)
	split_col = int(imgcols/num_regions)
        region_rows = [min(i*split_row, imgrows) for i in range(num_regions+1)]
        region_cols = [min(i*split_col, imgcols) for i in range(num_regions+1)]
	#if(region_rows[num_regions] < imgrows):
	#	region_rows.append(imgrows)
	#if(region_cols[num_regions] < imgcols):
	#	region_cols.append(imgcols)
	return [region_rows, region_cols]

def _find_min_max_sub(subset):
	subset = subset.ravel()
	return [np.amin(subset), np.amax(subset)]

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
	
def calculate_edge_density(wsize, img):
	#wsize refers to window size
	print 'Started edge density calculation'
	imgshp = img.shape
	#print imgshp
	img1 = np.zeros(imgshp)
	img1 = img1.astype(np.float)
	side = 2*wsize
	denom = side*side 
	for i in xrange(wsize, imgshp[0]-wsize-1):
		start = time.time()
		for j in xrange(wsize, imgshp[1]-wsize-1):
			imin = i - wsize
			imax = i + wsize 
			jmin = j - wsize
			jmax = j + wsize 
			current = img[imin:imax, jmin: jmax].ravel()
			#print current.shape
			img1[i,j] = np.sum(current)/denom
		#if(i%1000==0):
			#print 'Finished edge density for wsize= %d, lines = %d, time = %s seconds ' %(2*wsize+1,i, time.time() - start)
	return(img1)

def canny_filter(t1,t2, sobelF, wsize):	
	ipath = '/scratch/slums/bl-slums/raw-img/MUL_mosaic_415.tif'
	[i1, imgrows, imgcols, imgbands, imggeotrans] = read_image(ipath)
	print(imgrows, imgcols)
	roi_rows, roi_cols = _generate_regions_of_interest_indices(imgrows, imgcols, NUM_REGIONS)
	b = i1.GetRasterBand(1)		
	
	print(roi_rows, roi_cols)
	edges = np.zeros((imgcols, imgrows), dtype = np.float)
	imgMin = np.inf
	imgMax = -np.inf
	for i in range(len(roi_rows)-1):
		#print b.XSize, b.YSize
		curr_min_row = roi_rows[i]
		curr_max_row = roi_rows[i+1]
			
		for j in range(len(roi_cols) -1):
			curr_min_col = roi_cols[j]
			curr_max_col = roi_cols[j+1]
			curr_row_size = curr_max_row - curr_min_row
			curr_col_size = curr_max_col - curr_min_col
			img = b.ReadAsArray(roi_rows[i],roi_cols[j], curr_row_size ,curr_col_size).astype(np.float)
			print 'Bounding box coordinates: [{},{}] to [{},{}]'.format(curr_min_row,curr_min_col, curr_max_row, curr_max_col)
			start = time.time()
			curMin, curMax = _find_min_max_sub(img)
			print("time taken to find min is %s seconds" %(time.time() - start))
			imgMin = min(curMin, imgMin)
			imgMax = max(curMax, imgMax)
			#print imgMin, imgMax
	k = 0
	print 'Total number of batches = {}'.format(len(roi_rows)*len(roi_cols))
	for i in range(len(roi_rows)-1):
		curr_min_row = roi_rows[i]
		curr_max_row = roi_rows[i+1]
	
		for j in range(len(roi_cols) -1):
			curr_min_col = roi_cols[j]
			curr_max_col = roi_cols[j+1]
			curr_row_size = curr_max_row - curr_min_row
			curr_col_size = curr_max_col - curr_min_col
			#print 'Current row offset: %d, Current col offset:  %d, Currrent xsize : %d, Current ysize = %d' %(curr_min_row, curr_min_col, curr_row_size, curr_col_size)
			img = b.ReadAsArray(roi_rows[i],roi_cols[j], curr_row_size ,curr_col_size).astype(np.float)
			imgTmp = np.ceil((img - imgMin)*255/(imgMax - imgMin))
			img =  imgTmp.astype(np.uint8)
			img = cv2.GaussianBlur(img,(5,5),0)	
			#print 'Completed gaussian blur'
			tmpImg = cv2.Canny(img, t1,t2,sobelF)
			start = time.time()
			edges[curr_min_col:curr_max_col, curr_min_row:curr_max_row] = calculate_edge_density(wsize, tmpImg)	
			#print('Completed edge density calculation for curr starting row = %d, Number of pixels = %d, time = %s seconds' %(curr_min_row, curr_row_size*curr_col_size, time.time() - start))
			k+=1
			print 'Finished {} batches, current row start = {}, current row end = {}, current col start = {}, current col end = {}'.format(k, curr_min_row, curr_max_row, curr_min_col, curr_max_col)
	return([imgrows, imgcols, imgbands, imggeotrans, edges])
	


if __name__ == "__main__":
	print 'Starting...'
	[imgrows, imgcols, imgbands, imggeotrans, edges] = canny_filter(30,50,5, 20)
	opath = '/scratch/slums/bl-slums/raw-img/MUL_mosaic_415_edgeDensity.tif'
	write_image(imgrows, imgcols, imgbands, imggeotrans,edges, opath)
	#print 'Finished canny...'
	#wsizes = [25]
	#a = Parallel(n_jobs = len(wsizes))(delayed(calculate_edge_density)(i, edges) for i in wsizes) 
