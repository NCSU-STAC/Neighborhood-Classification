'''
Detect changes (new informal settlements) by comparing VHR image from 2016 (informal settlement mask) against ETM-Landsat 7 image from 2007 (urban mask)
If there was no urban construction in 2007 at a location where it is identified as informal in 2016, note this as a change in the 2016 result
@author: Krishna Karthik Gadiraju
'''
from osgeo import gdal
import osr, ogr
import numpy as np
import time, sys
import pickle



def write_image(imgrows, imgcols, imgbands, imggeotrans,edges, opath):
    driver = gdal.GetDriverByName('GTiff')
    outRaster = driver.Create(opath, imgrows, imgcols, imgbands, gdal.GDT_Byte)
    outRaster.SetGeoTransform(imggeotrans)
    outband = outRaster.GetRasterBand(1)
    outband.WriteArray(edges)
    outRasterSRS = osr.SpatialReference()
    outRasterSRS.ImportFromEPSG(32643)
    outRaster.SetProjection(outRasterSRS.ExportToWkt())
    outband.FlushCache()

driver = gdal.GetDriverByName('GTiff')
#Classified VHR image (as IF vs all other classes)
rasterFileName1 = "/scratch/slums/bl-slums/change-detection/xgboost-3class-final-classification-clipped.tif"
dataset1 = gdal.Open(rasterFileName1)

rasterFileName2 = "/scratch/slums/bl-slums/change-detection/LS-VCLF-Feb-2002.tif"
dataset2 = gdal.Open(rasterFileName2)

bands1=[]
bands2=[]


# Get rows, columns and other metdata pertaining to the two rasters
cols1 = dataset1.RasterXSize
rows1 = dataset1.RasterYSize

cols2 = dataset2.RasterXSize
rows2 = dataset2.RasterYSize

transform1 = dataset1.GetGeoTransform()
xOrigin1 = transform1[0]
yOrigin1 = transform1[3]
pixelWidth1 = transform1[1]
pixelHeight1 = -transform1[5]

transform2 = dataset2.GetGeoTransform()
xOrigin2 = transform2[0]
yOrigin2 = transform2[3]
pixelWidth2 = transform2[1]
pixelHeight2 = -transform2[5]

# Extract band information
bands1.append(dataset1.GetRasterBand(1))
bands2.append(dataset2.GetRasterBand(1)) 

print 'VHR image: XOrigin: {}, YOrigin: {}, pixelWidth = {}, pixelHeight = {}'.format(xOrigin1, yOrigin1, pixelWidth1, pixelHeight1)

print 'ETM image: XOrigin: {}, YOrigin: {}, pixelWidth = {}, pixelHeight = {}'.format(xOrigin2, yOrigin2, pixelWidth2, pixelHeight2)

result_raster = np.zeros((rows1,cols1), dtype = np.int8)

print 'ETM resoultion = {} x {}, VHR resolution = {} x {}'.format(rows2,cols2, rows1, cols1)

etm_parser_check = np.zeros((rows2,cols2), dtype = np.int8)
for row in xrange(0,rows1, 30):
    etm_row = row/30
    for col in xrange(0,cols1, 30):
        etm_col = col/30
        if etm_row < rows2 and etm_col < cols2:
            etm_parser_check[etm_row, etm_col] = 1
            #print 'VHR row = {}, VHR col = {}, ETM row = {}, ETM col = {}'.format(row, col, etm_row, etm_col)
            # Read the 30 * 30 region for VHR and the single pixel for ETM from their corresponding rasters
            vhr_data = bands1[0].ReadAsArray(col,row,30,30)
            etm_data = bands2[0].ReadAsArray(etm_col,etm_row,1,1)

            # Check change detection
            # identify if there are any slum classifications in the 30*30 pixel
            slum_indices = np.where(vhr_data == 0)
            #print np.shape(slum_indices[0])
            # Verify only if it is classified as slum, otherwise ignore
            if (np.shape(vhr_data)[0] > 0):
                if(etm_data == 0): # we are only looking for those slum instances that are categorized as urban
                    for slum_row, slum_col in zip(slum_indices[0], slum_indices[1]):
                        result_raster[slum_row + row, slum_col+col] = 1
    if row > 0 and row %10000 == 0:
       print 'Finished {} rows'.format(row)

changed_pixels = np.sum(result_raster)
changed_pixels = float(changed_pixels)
total_pixels = float(rows1 * cols1)               
percentage_change = (changed_pixels*100)/total_pixels
print 'Number of changed locations = {}, total pixels = {}, percentage change = {}'.format(changed_pixels, total_pixels,percentage_change )
print 'Parsed pixels = {}, Total ETM pixels = {}'.format(np.sum(etm_parser_check), rows2*cols2)

write_image(cols1, rows1, 1,transform1, result_raster, '/scratch/slums/bl-slums/change-detection/xgboost-slum-urban-detected-changes.tif')               


