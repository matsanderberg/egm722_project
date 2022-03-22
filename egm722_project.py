import numpy as np
import rasterio as rio
import rasterio.mask as mask
import geopandas as gpd
import cartopy.crs as ccrs
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.axes_grid1 import make_axes_locatable
#import earthpy.plot as ep

class SatelliteImg:

    # Spectral band mapping data structure for different satellites
    bands = {'Sentinel': {'Blue': 2, 'Green': 3, 'Red': 4, 'NIR': 8, 'SWIR': 12},
             'Landsat8': {'Blue': 2, 'Green': 3, 'Red': 4}}

    img = {}
    extent = []

    def __init__(self, satellite, file_path, date):
        self.satellite = satellite
        self.file_path = file_path
        self.date = date

    def load_band_data(self, outline_fp=None):
        # Crop image to outline if provided
        if outline_fp is not None:
            outline = gpd.read_file(outline_fp)
            outline = outline.to_crs(epsg=32633)  # TODO: should match the dataset

            with rio.open(self.file_path) as src:
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
                    with rio.open("data_files/cropped_temp.tif", "w", **out_meta) as dest:
                        dest.write(out_image)
                        xmin, ymin, xmax, ymax = dest.bounds
                        self.file_path = dest.name  # update file path to dataset with cropped image

        # Load the bands depending on satellite
        with rio.open(self.file_path) as dataset:
            # read the spectral bands depending on satellite
            bands = self.bands[self.satellite]
            for key, band in bands.items():
                self.img[key] = dataset.read(band)

            xmin, ymin, xmax, ymax = dataset.bounds
            SatelliteImg.extent = [xmin, xmax, ymin, ymax]

    '''
    TODO: write docstring
    NBR = (NIR - SWIR)/(NIR + SWIR)
    '''
    def nbr(self):
        # Suppressing runtime warning for division by zero
        np.seterr(divide='ignore', invalid='ignore')
        nbr = (self.img['NIR'].astype(int) - self.img['SWIR'].astype(int)) / (self.img['NIR'].astype(int) + self.img['SWIR'].astype(int))

        # Replace NaN with 0
        np.nan_to_num(nbr, copy=False, nan=0)
        return nbr

    '''
    TODO: write docstring
    NBR = (NIR - RED)/(NIR + RED)
    '''
    def ndvi(self):
        # Suppressing runtime warning for division by zero
        np.seterr(divide='ignore', invalid='ignore')
        ndvi = (self.img['NIR'].astype(int) - self.img['Red'].astype(int)) / (self.img['NIR'].astype(int) + self.img['Red'].astype(int))

        # Replace NaN with 0
        np.nan_to_num(nbr, copy=False, nan=0)
        return ndvi

    def description(self):
        print('Satellite: ' + self.satellite)
        print('Image size: ' + str(self.img['Blue'].shape))
        print('Spectral bands: ')
        for key in self.img.keys():
            print(key)


pre_fire = SatelliteImg('Sentinel', 'data_files/karbole_sentinel2_june26.img', '2018-06-26')
pre_fire.load_band_data('data_files/outline.shp')
pre_nbr = pre_fire.nbr()

post_fire = SatelliteImg('Sentinel', 'data_files/karbole_sentinel2_aug2.img', '2018-08-23')
post_fire.load_band_data('data_files/outline.shp')
post_nbr = post_fire.nbr()

dnbr1 = pre_nbr - post_nbr

pre_fire.description()
post_fire.description()


def index(band1, band2):
    '''
    TODO: write docstring
    NBR = (NIR - SWIR2)/(NIR + SWIR2)
    NIR = Band 8
    SWIR2 = Band 12
    '''
    # Suppressing runtime warning for division by zero
    np.seterr(divide='ignore', invalid='ignore')

    nbr = (band1 - band2) / (band1 + band2)

    # Replace NaN with 0
    np.nan_to_num(nbr, copy=False, nan=0)

    return nbr

def delta(pre_img, post_img):
    '''
    TODO: write docstring
    NBR = (NIR - SWIR2)/(NIR + SWIR2)
    NIR = Band 8
    SWIR2 = Band 12
    '''
    return pre_img - post_img

def generate_handles(labels, colors, edge='k', alpha=1):
    lc = len(colors)  # get the length of the color list
    handles = []
    for i in range(len(labels)):
        handles.append(mpatches.Rectangle((0, 0), 1, 1, facecolor=colors[i % lc], edgecolor=edge, alpha=alpha))
    return handles

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

def load_band_data(file_path, band, outline):
    '''
    This is where you should write a docstring.
    '''
    with rio.open(file_path) as src:
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
            with rio.open("data_files/cropped_temp.tif", "w", **out_meta) as dest:
                dest.write(out_image)

            # Read cropped image to memory
            with rio.open("data_files/cropped_temp.tif") as dataset:
                img = dataset.read(band)
                xmin, ymin, xmax, ymax = dataset.bounds

            extent = [xmin, xmax, ymin, ymax]

            # Some debug printing during development
            #print('{} opened in {} mode'.format(dataset.name, dataset.mode))
            #print('image has {} band(s)'.format(dataset.count))
            #print('image size (width, height): {} x {}'.format(dataset.width, dataset.height))
            #print('band 1 dataype is {}'.format(dataset.dtypes[0]))  # note that the band name (Band 1) differs from the list index [0]

    return img, extent

# Read the outline to mask the raster with
outline = gpd.read_file('data_files/outline.shp')
outline = outline.to_crs(epsg=32633) # should match the dataset

prefire_path = 'data_files/karbole_sentinel2_june26.img' # TODO: read from init file
postfire_path = 'data_files/karbole_sentinel2_aug2.img' # TODO: read from init file

# Load band arrays
prefire_b4, extent = load_band_data(prefire_path, 4, outline)
prefire_b8a, extent = load_band_data(prefire_path, 8, outline)
prefire_b12, extent = load_band_data(prefire_path, 12, outline)
postfire_b4, extent = load_band_data(prefire_path, 4, outline)
postfire_b8a, extent = load_band_data(postfire_path, 8, outline)
postfire_b12, extent = load_band_data(postfire_path, 12, outline)

pre_ndvi = index(prefire_b8a.astype(int), postfire_b4.astype(int))
post_ndvi = index(postfire_b8a.astype(int), postfire_b4.astype(int))
dndvi = delta(pre_ndvi, post_ndvi)

pre_nbr = index(prefire_b8a.astype(int), prefire_b12.astype(int))
post_nbr = index(postfire_b8a.astype(int), postfire_b12.astype(int))
dnbr2 = delta(pre_nbr, post_nbr)

myCRS = ccrs.UTM(33) # note that this matches with the CRS of our image
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 10), subplot_kw=dict(projection=myCRS))

# Set colors for plotting and classes
cmap = matplotlib.colors.ListedColormap(['green','yellow','orange','red','purple'])
cmap.set_over('purple')
cmap.set_under('white')
bounds = [-0.5, 0.1, 0.27, 0.440, 0.660, 1.3] # dNBR threshold values as defined by UN-Spider
norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)
plt.title('Burn Severity Map')

# Legend
labels = ['Unburned', 'Low Severity', 'Moderate-low Severity', 'Moderate-high Severity', 'High Severity']
colors = ['green','yellow','orange','red','purple']
handles = generate_handles(labels, colors)
ax1.legend(handles, labels, fontsize=6, loc='lower left', framealpha=1)

# Plot
#h, ax = img_display(img, ax, [12,8,3], myCRS, [xmin, xmax, ymin, ymax])
h = ax1.imshow(dnbr1, cmap=cmap, norm=norm, transform=myCRS, extent=extent)
#divider = make_axes_locatable(ax)
#cax = divider.append_axes("right", size="5%", pad=0.1, axes_class=plt.Axes)
#fig.colorbar(h, cax=cax, label='NBR')

#cbar = fig.colorbar(h, ax=ax, fraction=0.035, pad=0.04, ticks=[-0.2, 0.18, 0.35, 0.53, 1])
#cbar.ax.set_yticklabels(['Unburned', 'Low Severity', 'Moderate-low Severity', 'Moderate-high Severity', 'High Severity'])

#bounds = [-0.1, 0.1, 0.2, 0.3, 0.5, 0.9] # dNBR threshold values as defined by UN-Spider
#norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)
#h = ax2.imshow(pre_ndvi, cmap=cmap, norm=norm, transform=myCRS, extent=extent)
#h = ax2.imshow(dndvi, cmap='Greys', vmin=-1, vmax=1, transform=myCRS, extent=extent)
h = ax2.imshow(dnbr2, cmap=cmap, norm=norm, transform=myCRS, extent=extent)
# save the figure
fig.savefig('output_maps/dnbr_dndvi.png', dpi=300, bbox_inches='tight')

# save the figure
#fig.savefig('output_maps/dndvi.png', dpi=300, bbox_inches='tight')




