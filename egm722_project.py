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
from mpl_toolkits.axes_grid1 import make_axes_locatable
#import earthpy.plot as ep

class SatelliteImg:

    # Spectral band mapping data structure for different satellites
    bands = {'Sentinel': {'Blue': 2, 'Green': 3, 'Red': 4, 'NIR': 9, 'SWIR': 11, 'SWIR2': 12},
             'Landsat8': {'Blue': 2, 'Green': 3, 'Red': 4}}

    def __init__(self, satellite, file_path, date):
        self.satellite = satellite
        self.file_path = file_path
        self.date = date
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

        # Load the bands depending on satellite
        with rio.open(self.file_path) as dataset:
            self.crs = dataset.crs
            self.transform = dataset.transform

            # Load selected spectral bands depending on satellite
            bands = self.bands[self.satellite]
            for key, band in bands.items():
                self.img[key] = dataset.read(band).astype(np.float32)

            # Load all the bands
            self.bands_data = dataset.read().astype(np.float32)

            xmin, ymin, xmax, ymax = dataset.bounds
            self.extent = [xmin, xmax, ymin, ymax]

        # Delete temp cropped file
        os.remove(self.file_path)

    def nbr(self):
        '''
        TODO: write docstring
        Normalized Burn Ratio
        NBR = (NIR - SWIR2)/(NIR + SWIR2)
        '''
        # Suppressing runtime warning for division by zero
        np.seterr(divide='ignore', invalid='ignore')
        return (self.img['NIR'] - self.img['SWIR2']) / (self.img['NIR'] + self.img['SWIR2'])

    def ndvi(self):
        '''
        TODO: write docstring
        Normalized Difference Vegetation Index
        NDVI = (NIR - RED)/(NIR + RED)
        '''
        # Suppressing runtime warning for division by zero
        np.seterr(divide='ignore', invalid='ignore')
        return (self.img['NIR'] - self.img['Red']) / (self.img['NIR'] + self.img['Red'])

    def ndmi(self):
        '''
        TODO: write docstring
        Normalized Difference Moisture Index
        NDVI = (NIR - SWIR)/(NIR + SWIR)
        '''
        # Suppressing runtime warning for division by zero
        np.seterr(divide='ignore', invalid='ignore')
        return (self.img['NIR'] - self.img['SWIR']) / (self.img['SWIR'] + self.img['Red'])

    def get_extent(self):
        return self.extent

    def description(self):
        print('Satellite: ' + self.satellite)
        print('Image size: ' + str(self.img['Blue'].shape))
        print('Spectral bands: ')
        for key in self.img.keys():
            print(key)

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

# Load data and calculate spectral indices
# -------------------------------------------------------------------------------------#
pre_fire = SatelliteImg('Sentinel', 'data_files/karbole_sentinel2_june26.img', '2018-06-26')
pre_fire.load_band_data('data_files/outline.shp')

post_fire = SatelliteImg('Sentinel', 'data_files/karbole_sentinel2_aug2.img', '2018-08-23')
post_fire.load_band_data('data_files/outline.shp')

pre_nbr = pre_fire.nbr()
post_nbr = post_fire.nbr()
dnbr = pre_nbr - post_nbr

pre_ndvi = pre_fire.ndvi()
post_ndvi = post_fire.ndvi()
dndvi = pre_ndvi - post_ndvi

pre_ndmi = pre_fire.ndmi()
post_ndmi = post_fire.ndmi()

extent = pre_fire.get_extent()

# Plotting
# --------------------------------------------------------------------------------------- #
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

# Set colors for plotting and classes for dNBR
fig, ax = plt.subplots(1, 1, figsize=(10, 10), subplot_kw=dict(projection=myCRS))

cmap = matplotlib.colors.ListedColormap(['green','yellow','orange','red','purple'])
cmap.set_over('purple') # sets the color for high out-of-range values
cmap.set_under('white') # sets the color for low out-of-range values
bounds = [-0.5, 0.1, 0.27, 0.440, 0.660, 1.3] # dNBR threshold values as defined by UN-SPIDER
norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)

# Another (more elaborate) way of creating the color bar
#divider = make_axes_locatable(ax)
#cax = divider.append_axes("right", size="5%", pad=0.1, axes_class=plt.Axes)
#cbar = fig.colorbar(h, ax=ax, fraction=0.035, pad=0.04, ticks=[-0.2, 0.18, 0.35, 0.53, 1])
#cbar.ax.set_yticklabels(['Unburned', 'Low Severity', 'Moderate-low Severity', 'Moderate-high Severity', 'High Severity'])

# Legend
labels = ['Unburned', 'Low Severity', 'Moderate-low Severity', 'Moderate-high Severity', 'High Severity']
colors = ['green', 'yellow', 'orange', 'red', 'purple']
handles = generate_handles(labels, colors)
ax.legend(handles, labels, fontsize=10, loc='lower left', framealpha=1)

h = ax.imshow(dnbr, cmap=cmap, norm=norm, transform=myCRS)
ax1.set_title("dNBR")

# Save the dNBR map
fig.savefig('output_maps/dnbr.png', dpi=300, bbox_inches='tight')


# Supervised Learning with Random Forest
# Based on tutorial from: https://adaneon.com/image-analysis-tutorials/pages/part_four.html
# ----------------------------------------------------------------------------------------

# Load data
dataset = SatelliteImg('Sentinel', 'data_files/karbole_sentinel2_aug2.img', '2018-08-23')
dataset.load_band_data('data_files/outline.shp')
training_data = gpd.read_file('data_files/burn_training.shp').to_crs(dataset.crs)

classifier = init_random_forest(dataset, training_data, 'Classvalue')
classification = random_forest(classifier, dataset)

#print("The three classes are: " + str(classes))
#print("Total number of training labels: " + str(training_labels.size))
#print("Total number of training sample size: " + str(training_samples.size))

# Plot classification image and source RGB
fig, (ax1, ax2) = plt.subplots(1,2, figsize =(18,15), subplot_kw=dict(projection=myCRS))
print(dataset.bands_data.shape)
#ax1.imshow(dataset.bands_data[4, 3, 2], transform=myCRS, extent=extent)
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
