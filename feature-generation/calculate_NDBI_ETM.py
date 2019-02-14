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

def ndbi_per_pixel(nir_val, swir_val):
	return (swir_val - nir_val) /(swir_val + nir_val)	

def generate_NDBI():	
	ipath = '/scratch/slums/bl-slums/raw-img/PS-feb-2002-extract.tif'
	[i1, imgrows, imgcols, imgbands, imggeotrans] = read_image(ipath)
	print(imgrows, imgcols)
        ndvi = np.zeros((imgcols, imgrows), dtype = np.float)
	print imgbands
        f = np.vectorize(ndbi_per_pixel)
        nir_b = i1.GetRasterBand(3)		
        swir_b = i1.GetRasterBand(4)
	nir = nir_b.ReadAsArray(0,0, imgrows ,imgcols).astype(np.float) 
	swir = swir_b.ReadAsArray(0,0, imgrows ,imgcols).astype(np.float) 
        numerator = nir - swir
	denominator = nir + swir
        denominator[denominator==0] = 1
        ndvi = numerator/denominator
	return([imgrows, imgcols, imgbands, imggeotrans, ndvi])



if __name__ == "__main__":
	print 'Starting...'
	[imgrows, imgcols, imgbands, imggeotrans, edges] = generate_NDBI()
	opath = '/scratch/slums/bl-slums/raw-img/ETM-feb-2002-extract-NDBI.tif'
	write_image(imgrows, imgcols,imgbands, imggeotrans,edges, opath)
