import numpy as np
import rasterio as rio
import rasterio.mask as mask
import rasterio.features
import geopandas as gpd
import cartopy.crs as ccrs
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn import metrics
from scipy.cluster.vq import *
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestClassifier
import os
import sys
from mpl_toolkits.axes_grid1 import make_axes_locatable
import config as cfg
import re
import datetime
from datetime import date

class SatelliteImg:

    # Spectral band mapping data structure for different satellites
    bands = cfg.bands

    def __init__(self, file_path):
        self.file_path = file_path
        self.crs = ""
        self.transform = ""
        self.img = {}
        self.bands_data = []
        self.extent = []

    def load_band_data(self, outline_fp=None):
        # Crop image to outline if provided
        if outline_fp is not None:
            with rio.open(self.file_path) as src:
                try:
                    outline = gpd.read_file(outline_fp).to_crs(src.crs)
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
                        self.file_path = dest.name  # update file path to dataset with cropped image

        with rio.open(self.file_path) as dataset:
            self.crs = dataset.crs
            self.transform = dataset.transform

            # Load all the bands
            self.bands_data = dataset.read().astype(np.float32)

            xmin, ymin, xmax, ymax = dataset.bounds
            self.extent = [xmin, xmax, ymin, ymax]

        # Delete temp cropped file
        os.remove("data_files/cropped_temp.tif")

    def nbr(self):
        '''
        TODO: write docstring
        Normalized Burn Ratio
        NBR = (NIR - SWIR2)/(NIR + SWIR2)
        '''
        # Suppressing runtime warning for division by zero
        np.seterr(divide='ignore', invalid='ignore')

        nir = self.bands_data[self.bands['NIR']-1]
        swir2 = self.bands_data[self.bands['SWIR2']-1]
        return (nir - swir2) / (nir + swir2)

    def ndvi(self):
        '''
        TODO: write docstring
        Normalized Difference Vegetation Index
        NDVI = (NIR - RED)/(NIR + RED)
        '''
        # Suppressing runtime warning for division by zero
        np.seterr(divide='ignore', invalid='ignore')

        nir = self.bands_data[self.bands['NIR'] - 1]
        red = self.bands_data[self.bands['red'] - 1]
        return (nir - red) / (nir + red)

    def ndmi(self):
        '''
        TODO: write docstring
        Normalized Difference Moisture Index
        NDMI = (NIR - SWIR)/(NIR + SWIR)
        '''
        # Suppressing runtime warning for division by zero
        np.seterr(divide='ignore', invalid='ignore')

        nir = self.bands_data[self.bands['NIR'] - 1]
        swir = self.bands_data[self.bands['SWIR'] - 1]
        return (nir - swir) / (nir + swir)

    def get_extent(self):
        return self.extent

    def description(self):
        print('Satellite: ' + self.satellite)
        print('image size (width, height): {} x {}'.format(self.bands_data.width, self.bands_data.height))
        print('Number of bands: ' + str(self.bands_data.count))

def generate_handles(labels, colors, edge='k', alpha=1):
    '''
    TODO: This is where you should write a docstring.
    '''
    lc = len(colors)  # get the length of the color list
    handles = []
    for i in range(len(labels)):
        handles.append(mpatches.Rectangle((0, 0), 1, 1, facecolor=colors[i % lc], edgecolor=edge, alpha=alpha))
    return handles

def img_display(image, ax, bands, transform, extent):
    '''
    TODO: This is where you should write a docstring.
    '''
    # first, we transpose the image to re-order the indices
    dispimg = image.transpose([1, 2, 0])

    # next, we have to scale the image.
    dispimg = dispimg / dispimg.max()

    # finally, we display the image
    handle = ax.imshow(dispimg[:, :, bands], transform=transform, extent=extent)

    return handle, ax

def plot_dnbr(dnbr, date, crs):
    fig, ax = plt.subplots(1, 1, figsize=(10, 10), subplot_kw=dict(projection=crs))

    # Set colors for plotting and classes for dNBR
    cmap = matplotlib.colors.ListedColormap(['green', 'yellow', 'orange', 'red', 'purple'])
    cmap.set_over('purple')  # sets the color for high out-of-range values
    cmap.set_under('white')  # sets the color for low out-of-range values
    bounds = [-0.5, 0.1, 0.27, 0.440, 0.660, 1.3]  # dNBR threshold values as defined by UN-SPIDER
    norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)

    # Another (more elaborate) way of creating the color bar
    # divider = make_axes_locatable(ax)
    # cax = divider.append_axes("right", size="5%", pad=0.1, axes_class=plt.Axes)
    # cbar = fig.colorbar(h, ax=ax, fraction=0.035, pad=0.04, ticks=[-0.2, 0.18, 0.35, 0.53, 1])
    # cbar.ax.set_yticklabels(['Unburned', 'Low Severity', 'Moderate-low Severity', 'Moderate-high Severity', 'High Severity'])

    # Legend
    labels = ['Unburned', 'Low Severity', 'Moderate-low Severity', 'Moderate-high Severity', 'High Severity']
    colors = ['green', 'yellow', 'orange', 'red', 'purple']
    handles = generate_handles(labels, colors)
    ax.legend(handles, labels, fontsize=10, loc='lower left', framealpha=1)

    ax.imshow(dnbr, cmap=cmap, norm=norm, transform=crs)
    ax.set_title("dNBR")

    # Save the dNBR map
    # Create result directory
    fig.savefig('output_maps/dnbr_' + date + '.png', dpi=300, bbox_inches='tight')

def reclassify_dbnr(dbnr):
    '''
    TODO: This is where you should write a docstring.
    Reclassifies the dnbr to burned or unburned
    '''
    reclassified_dnbr = np.zeros((dnbr.shape[0], dnbr.shape[1]))
    for i in range(0, dnbr.shape[0]):
        for j in range(0, dnbr.shape[1]):
            if dnbr[i][j] < 0.1:    # Unburned
                reclassified_dnbr[i][j] = 2
            else:                   # Burnt
                reclassified_dnbr[i][j] = 1

    return reclassified_dnbr

def init_random_forest(dataset, training_data, label):
    '''
    TODO: This is where you should write a docstring.
    Reclassifies the dnbr to burned or unburned
    '''
    n_bands, rows, cols = dataset.bands_data.shape
    #classes = training_data[label].unique()

    shapes = list(zip(training_data['geometry'], training_data[label]))
    labeled_pixels = rio.features.rasterize(shapes=shapes, out_shape=(rows, cols), fill=0, transform=dataset.transform)

    # Plot the rasterized output
    fig = plt.figure()
    plt.imshow(labeled_pixels)
    fig.savefig('output_maps/training_data.png', dpi=300, bbox_inches='tight')

    # Filter non-zero values
    is_train = np.nonzero(labeled_pixels)

    # Get label and sample data
    training_labels = labeled_pixels[is_train]
    bands_data = np.dstack(dataset.bands_data)
    training_samples = bands_data[is_train]

    # Dispatch computation on all the CPUs
    classifier = RandomForestClassifier(n_jobs=-1)

    # Fit the classifier
    classifier.fit(training_samples, training_labels)

    return classifier

def random_forest(classifier, dataset):
    '''
    TODO: This is where you should write a docstring.
    '''
    n_bands, rows, cols = dataset.bands_data.shape
    # Resampling size
    n_samples = rows * cols

    bands_data = np.dstack(dataset.bands_data)
    # Reshape the dimension
    flat_pixels = bands_data.reshape((n_samples, n_bands))

    # Make prediction
    result = classifier.predict(flat_pixels)

    # Reshape output two dimension
    classification = result.reshape((rows, cols))

    return classification

# dNBR analysis on available data
# -------------------------------------------------------------------------------------#
postfire_rasters = []
prefire_raster = ''
files = []
fire_boundary = None
crs = ccrs.UTM(33) # TODO: this should match with the CRS of our image

# Iterate the data directory specified in config.py and extract the needed files
try:
    files = os.listdir(cfg.data_dir)
except FileNotFoundError:
    sys.exit("No files found in the filepath specified by config.py")

for f in files:
    if f.endswith('boundary.shp'):
        fire_boundary = os.path.join(cfg.data_dir, f)
    elif f.endswith('prefire.img'):
        prefire_raster = os.path.join(cfg.data_dir, f)
    else:
        # Find all post fire rasters. Need to end with 8 digits
        match = bool(re.search(r'\d{8}.img', f))
        if match:
            postfire_rasters.append(os.path.join(cfg.data_dir, f))

# Run the dNBR analysis with the available data (but only of data is available)
if prefire_raster and postfire_rasters:
    pre_fire = SatelliteImg(prefire_raster)
    pre_fire.load_band_data(fire_boundary)
    pre_nbr = pre_fire.nbr()

    # Loop post fire images
    for file in postfire_rasters:
        post_fire = SatelliteImg(file)
        post_fire.load_band_data(fire_boundary)
        post_nbr = post_fire.nbr()
        dnbr = pre_nbr - post_nbr

        # Plot and save data
        # Todo: date
        plot_dnbr(dnbr, '20180903', crs)

        # Todo: calculate statistics
        # Todo: save to datebase?
else:
    sys.exit("No data matching the required file pattern found in specified folder.")


# Calculating NDVI and NDMI indices pre and post fire
# --------------------------------------------------------------------------------------- #
pre_ndvi = pre_fire.ndvi()
post_ndvi = post_fire.ndvi()
dndvi = pre_ndvi - post_ndvi

pre_ndmi = pre_fire.ndmi()
post_ndmi = post_fire.ndmi()

extent = pre_fire.get_extent()

myCRS = ccrs.UTM(33) # TODO: this should match with the CRS of our image
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16), subplot_kw=dict(projection=myCRS))

h = ax1.imshow(pre_ndmi, cmap='PuBuGn')
fig.colorbar(h, ax=ax1)
ax1.set_title("Pre Fire NDMI")
h = ax2.imshow(post_ndmi, cmap='PuBuGn')
fig.colorbar(h, ax=ax2)
ax2.set_title("Post Fire NDMI")
h = ax3.imshow(pre_ndvi, cmap='RdYlGn')
fig.colorbar(h, ax=ax3)
ax3.set_title("Pre Fire NDVI")
h = ax4.imshow(post_ndvi, cmap='RdYlGn')
fig.colorbar(h, ax=ax4)
ax4.set_title("Post Fire NDVI")

# Save the figure
fig.savefig('output_maps/ndmi_ndvi.png', dpi=300, bbox_inches='tight')

# Supervised Learning with Random Forest
# Based on tutorial from: https://adaneon.com/image-analysis-tutorials/pages/part_four.html
# ----------------------------------------------------------------------------------------

# Load data
dataset = SatelliteImg('data_files/karbole_sentinel2_20180802.img')
dataset.load_band_data('data_files/fire_boundary.shp')
training_data = gpd.read_file('data_files/training_data.shp').to_crs(dataset.crs)

# Initialize and run the classifier
classifier = init_random_forest(dataset, training_data, 'Classvalue')
classification = random_forest(classifier, dataset)

#print("The three classes are: " + str(classes))
#print("Total number of training labels: " + str(training_labels.size))
#print("Total number of training sample size: " + str(training_samples.size))

# Plot classification image and source RGB
fig, (ax1, ax2) = plt.subplots(1,2, figsize =(18,15), subplot_kw=dict(projection=myCRS))
img_display(dataset.bands_data, ax1, [7, 3, 2], myCRS, extent)
ax2.imshow(classification, transform=myCRS, extent=extent)
fig.savefig('output_maps/classification.png', dpi=300, bbox_inches='tight')

# Calculate size of area and compare to dNBR
burned = classification[classification == 1]
burned_size = (burned.size*20*20)/1000000
unburned = classification[classification == 2]
water = classification[classification == 3]
unburned_size = ((water.size + unburned.size)*20*20)/1000000
print("With Random Forest classification")
print("Burned area (km2): " + str(burned_size))
print("Unburned area (km2): " + str(unburned_size))

reclass_dnbr = reclassify_dbnr(dnbr)
burned = reclass_dnbr[reclass_dnbr == 1]
burned_size = (burned.size*20*20)/1000000
unburned = reclass_dnbr[reclass_dnbr == 2]
unburned_size = (unburned.size*20*20)/1000000
print("With dNBR")
print("Burned area (km2): " + str(burned_size))
print("Unburned area (km2): " + str(unburned_size))
