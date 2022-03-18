import numpy as np
import rasterio as rio
import rasterio.mask as mask
import geopandas as gpd
import cartopy.crs as ccrs
import matplotlib.pyplot as plt


def img_display(image, ax, bands, transform, extent):
    '''
    This is where you should write a docstring.
    '''
    # first, we transpose the image to re-order the indices
    dispimg = image.transpose([1, 2, 0])

    # next, we have to scale the image.
    dispimg = dispimg / dispimg.max()

    # finally, we display the image
    handle = ax.imshow(dispimg[:, :, bands], transform=transform, extent=extent)

    return handle, ax

# Read the outline to mask the raster with
outline = gpd.read_file('data_files/outline.shp')
outline = outline.to_crs(epsg=32633) # should match the dataset

# Open the src raster and crop it to outline
# TODO: convert to function and read src and outline from file
with rio.open('data_files/karbole_sentinel2_june26.img') as src:
    out_image, out_transform = mask.mask(src, outline['geometry'], crop=True)
    out_meta = src.meta

    out_meta.update({"driver": "GTiff",
                    "height": out_image.shape[1],
                    "width": out_image.shape[2],
                    "transform": out_transform})

    # Write the cropped image to file
    with rio.open("data_files/study_area.tif", "w", **out_meta) as dest:
        dest.write(out_image)

    # Read cropped image to memory
    with rio.open("data_files/study_area.tif") as dataset:
        img = dataset.read()
        xmin, ymin, xmax, ymax = dataset.bounds

print('{} opened in {} mode'.format(dataset.name, dataset.mode))
print('image has {} band(s)'.format(dataset.count))
print('image size (width, height): {} x {}'.format(dataset.width, dataset.height))
print('band 1 dataype is {}'.format(dataset.dtypes[0])) # note that the band name (Band 1) differs from the list index [0]

myCRS = ccrs.UTM(33) # note that this matches with the CRS of our image
fig, ax = plt.subplots(1, 1, figsize=(10, 10), subplot_kw=dict(projection=myCRS))
h, ax = img_display(img, ax, [2, 1, 0], myCRS, [xmin, xmax, ymin, ymax])
# save the figure
fig.savefig('output_maps/map.png', dpi=300, bbox_inches='tight')




