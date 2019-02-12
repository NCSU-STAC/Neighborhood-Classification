
# coding: utf-8

'''
Code for generating single raster from all the smaller rasters generated during distributed classification
Given that the final bangalore raster is ~94000*94000 pixels, the entire image wouldn't fit in memory
We divide the image into sections and classify each one separately and merge the final sub-rasters using this file
@author: Krishna Karthik Gadiraju/kkgadiraju
'''


#### Read the config for the original PS file to be used to create final Geotiff file


# In[ ]:


from osgeo import gdal
import osr,ogr
import numpy as np
import fiona

driver = gdal.GetDriverByName('GTiff')
#rasterFileName = "/home/gadiraju/data/bl-slums/raw-img/MUL_mosaic_415.tif" #path to raster
rasterFileName1 = "/scratch/slums/bl-slums/raw-img/PS_mosaic_415.tif"
dataset1 = gdal.Open(rasterFileName1)
cols = dataset1.RasterXSize
rows = dataset1.RasterYSize

transform = dataset1.GetGeoTransform()

print '{}'.format(transform)


# In[ ]:


#### Read each matrix from disk
# splits = [[0,15000],[15000,30000],[30000, 40000],[40000,50000],[50000,60000],[60000, 70000], [70000, 80000], [80000, 94321]]
splits = []
for i in range(0, 94321, 250):
    if i+250 > 94321:
        splits.append([i, 94321])
    else:
        splits.append([i, i + 250])
print splits


# Give the name of your classifier here
clf ='rf'
for split in splits:
    print split
    curr_mat = np.load('/scratch/slums/bl-slums/2-class-matrices-from-arc/{}-clf-L1-{}-{}.bin.npy'.format(clf,split[0], split[1])).astype(np.int8)
    if split[0]==0:
        final_mat = curr_mat
    else:
        final_mat = np.vstack((final_mat, curr_mat))



# In[ ]:


def write_image(imgrows, imgcols, imgbands, imggeotrans,edges, opath):
    driver = gdal.GetDriverByName('GTiff')
    outRaster = driver.Create(opath, imgrows, imgcols, imgbands, gdal.GDT_Int16)
    outRaster.SetGeoTransform(imggeotrans)
    outband = outRaster.GetRasterBand(1)
    outband.WriteArray(edges)
    outRasterSRS = osr.SpatialReference()
    outRasterSRS.ImportFromEPSG(32643)
    outRaster.SetProjection(outRasterSRS.ExportToWkt())
    outband.FlushCache()

write_image(cols, rows, 1, transform,final_mat, '/scratch/slums/bl-slums/clf-img/{}-L1-final-classification.tif'.format(clf))
