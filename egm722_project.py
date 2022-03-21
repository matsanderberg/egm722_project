import numpy as np
import rasterio as rio
import rasterio.mask as mask
import geopandas as gpd
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
#import earthpy.plot as ep

def calculate_dnbr(image):
    '''
    TODO: write docstring
    NBR = (NIR - SWIR2)/(NIR + SWIR2)
    NIR = Band 8
    SWIR2 = Band 12
    '''
    # Suppressing runtime warning for division by zero
    np.seterr(divide='ignore', invalid='ignore')
    nbr = (image[8] - image[12])/(image[8] + image[12])

    # Replace NaN with 0
    np.nan_to_num(nbr, copy=False, nan=0)

    return nbr

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
# TODO: convert to function and read src, outline, bands from file
with rio.open('data_files/karbole_sentinel2_aug2.img') as src:
    try:
        out_image, out_transform = mask.mask(src, outline['geometry'], crop=True)
    except ValueError:
        print(f"No overlap found for {src.files[0]}.")
    else:
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

        # Some debug printing during development
        print('{} opened in {} mode'.format(dataset.name, dataset.mode))
        print('image has {} band(s)'.format(dataset.count))
        print('image size (width, height): {} x {}'.format(dataset.width, dataset.height))
        print('band 1 dataype is {}'.format(dataset.dtypes[0])) # note that the band name (Band 1) differs from the list index [0]

myCRS = ccrs.UTM(33) # note that this matches with the CRS of our image
fig, ax = plt.subplots(1, 1, figsize=(10, 10), subplot_kw=dict(projection=myCRS))
#calculate_dnbr(img)
#h, ax = img_display(img, ax, [12,8,3], myCRS, [xmin, xmax, ymin, ymax])

nbr = calculate_dnbr(img)
h = ax.imshow(nbr, cmap='Greys', vmin=-1, vmax=1, transform=myCRS, extent=[xmin, xmax, ymin, ymax])
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.1, axes_class=plt.Axes)
fig.colorbar(h, cax=cax, label='NBR')
             #legend=True, cax=cax, legend_kwds={'label': 'NBR'})

# save the figure
fig.savefig('output_maps/map.png', dpi=300, bbox_inches='tight')




