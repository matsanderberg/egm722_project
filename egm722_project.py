%matplotlib notebook

import numpy as np
import rasterio as rio
import geopandas as gpd
import matplotlib.pyplot as plt
from rasterstats import zonal_stats

# Open the post fire raster and read the data
with rio.open('data_files/LCM2015_Aggregate_100m.tif') as dataset:
    xmin, ymin, xmax, ymax = dataset.bounds
    crs = dataset.crs
    landcover = dataset.read(1)
    width = dataset.width
    height = dataset.height
    affine_tfm = dataset.transform