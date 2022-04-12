import os
import sys
import re
import datetime
from datetime import date
import numpy as np
import rasterio as rio
import rasterio.mask as mask
import rasterio.features
import geopandas as gpd
import cartopy.crs as ccrs
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.ensemble import RandomForestClassifier
from mpl_toolkits.axes_grid1 import make_axes_locatable
import config as cfg

class SatelliteImg:
    '''
    SatelliteImg holds the band data for a given satellite image and has methods to
    calculate different spectral indices.

    Attributes:
        file_path: string holding the path to the satellite image
        date: the date of the image.
        TODO: remove these?
        crs: string for the coordinate referense system for the image. Updated when load_band_data is called
        transform: string for the transform of the image. Updated when load_band_data is called
        bands_data: a 3D list holding the band date for each band. Updated when load_band_data is called
        extent: a list holding the extent of the image (width, height)
    '''

    # Spectral band mapping depending on satellite
    bands = cfg.bands

    def __init__(self, file_path, date):
        self.file_path = file_path
        self.date = date
        self.crs = ""
        self.transform = ""
        self.bands_data = []
        self.extent = []

    def load_band_data(self, fire_boundary=None):
        '''
        Loads all band data into a 3D list. If a boundary shape file is provided the
        function will crop the image to that extent.

        Args:
            fire_boundary: file path to boundary shape file
        '''
        # Crop image to outline if provided
        if fire_boundary is not None:
            with rio.open(self.file_path) as src:
                try:
                    boundary = gpd.read_file(fire_boundary).to_crs(src.crs)
                    out_image, out_transform = mask.mask(src, boundary['geometry'], crop=True)
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
            print("Image loaded:")
            print(self.description())

            xmin, ymin, xmax, ymax = dataset.bounds
            self.extent = [xmin, xmax, ymin, ymax]

        # Delete temp cropped file
        if os.path.exists(cfg.data_dir + "cropped_temp.tif"):
            os.remove(cfg.data_dir + "cropped_temp.tif")

    def nbr(self):
        '''
        Calculates the Normalized Burn Ratio (NBR) as
        NBR = (NIR - SWIR2)/(NIR + SWIR2)

        Returns:
            A list with NBR values for the satellite image
        '''
        print("Calculating NBR for " + self.file_path);
        # Suppressing runtime warning for division by zero
        np.seterr(divide='ignore', invalid='ignore')

        nir = self.bands_data[self.bands['NIR']-1]
        swir2 = self.bands_data[self.bands['SWIR2']-1]
        return (nir - swir2) / (nir + swir2)

    def ndvi(self):
        '''
        Calculates the Normalized Difference Vegetation Index (NDVI) as
        NDVI = (NIR - RED)/(NIR + RED)

        Returns:
            A list with NDVI values for the satellite image
        '''
        # Suppressing runtime warning for division by zero
        np.seterr(divide='ignore', invalid='ignore')

        nir = self.bands_data[self.bands['NIR'] - 1]
        red = self.bands_data[self.bands['red'] - 1]
        return (nir - red) / (nir + red)

    def ndmi(self):
        '''
        Calculates the Normalized Difference Moisture Index (NDMI) as
        NDMI = (NIR - SWIR)/(NIR + SWIR)

        Returns:
            A list with NDVI values for the satellite image
        '''
        # Suppressing runtime warning for division by zero
        np.seterr(divide='ignore', invalid='ignore')

        nir = self.bands_data[self.bands['NIR'] - 1]
        swir = self.bands_data[self.bands['SWIR'] - 1]
        return (nir - swir) / (nir + swir)

    def get_extent(self):
        '''
        Returns: extent of satellite image as a list [width, height]
        '''
        return self.extent

    def description(self):
        '''
        Prints a description of the satellite image
        '''
        print('Image: ' + self.file_path)
        print('Date: ' + str(self.date))
        print('(bands, width, height): ' + str(self.bands_data.shape))

def generate_handles(labels, colors, edge='k', alpha=1):
    '''
    Generates a list of matplot Rectangle handles for the legend.

    Args:
        labels: list of label names
        colors: list of colors
        edge: edge color. Defaults to k.
        alpha: blending value, 0-1. Defaults to 1.

    Returns:
        A list of handles
    '''
    lc = len(colors)  # get the length of the color list
    handles = []
    for i in range(len(labels)):
        handles.append(mpatches.Rectangle((0, 0), 1, 1, facecolor=colors[i % lc], edgecolor=edge, alpha=alpha))
    return handles

def img_display(image, ax, bands, transform, extent):
    '''
    Plots a scaled satellite image for the given band combination

    Args:
        image: 3D list for the satellite image to display
        ax: the matplot axes on which to plot image
        bands: the band combination to display. E.g. [7,3,2] for a Sentinel 2 IR combination
        transform: the image transform
        extent: the image extent

    Returns:
        handle: the AxesImage object
        ax: the matplotlib axes
    '''
    # first, we transpose the image to re-order the indices
    dispimg = image.transpose([1, 2, 0])

    # next, we have to scale the image.
    dispimg = dispimg / dispimg.max()

    # finally, we display the image
    handle = ax.imshow(dispimg[:, :, bands], transform=transform, extent=extent)

    return handle, ax

def dnbr(prefire, postfire):
    '''
    Takes two SatelliteImg objects and calculates the NBR delta

    Returns:
        dNBR for the two input images
    '''
    return prefire.nbr() - postfire.nbr()

def dndvi(prefire, postfire):
    '''
    Takes two SatelliteImg objects and calculates the NDVI delta

    Returns:
        dNDVI for the two input images
    '''
    return prefire.ndvi() - postfire.ndvi()

def plot_dnbr(dnbr, date, crs):
    '''
    Plots a burn severity map for DNBR and saves to file (a directory will be created
    named result in script root directory).

    Args:
        dnbr: the dNBR to plot
        date: date to be used in plot title
        crs: the projection needed for plotting
    '''
    #fig, ax = plt.subplots(1, 1, figsize=(10, 10), subplot_kw=dict(projection=crs))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 12), subplot_kw=dict(projection=crs))

    # Set colors for plotting and classes for dNBR
    cmap = matplotlib.colors.ListedColormap(cfg.colors)
    bounds = cfg.bounds
    norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)

    # Legend
    labels = cfg.labels
    handles = generate_handles(labels, cfg.colors)
    ax1.legend(handles, labels, fontsize=10, loc='lower left', framealpha=1)

    ax1.imshow(dnbr, cmap=cmap, norm=norm, transform=crs)
    ax1.set_title("Burn severity map with dNBR " + cfg.name + ", Date: " + str(date))

    # Pie chart for each class of burn severity
    sizes = [20, 20, 20, 20, 20]
    data = [234, 4325, 1212, 212, 245]

    ax2.pie(data, autopct='%1.1f%%', labels=labels, colors=cfg.colors)
    ax2.axis('equal')

    # Save the dNBR map
    # Todo: Create result directory
    fig.savefig('output_maps/dnbr_' + cfg.name + '_' + str(date) + '.png', dpi=300, bbox_inches='tight')

def reclassify_dbnr(dbnr, threshold):
    '''
    Reclassifies the dnbr to either burned or unburned for the given threshold

    Args:
        threshold: the delimiter value for burned or unburned area

    Returns:
        The reclassified dNBR
    '''
    reclassified_dnbr = np.zeros((dnbr.shape[0], dnbr.shape[1]))
    for i in range(0, dnbr.shape[0]):
        for j in range(0, dnbr.shape[1]):
            if dnbr[i][j] < threshold:    # Unburned
                reclassified_dnbr[i][j] = 2
            else:                   # Burnt
                reclassified_dnbr[i][j] = 1

    return reclassified_dnbr

def init_random_forest(dataset, training_data, label):
    '''
    Initialises a skitlearn Random Forest classifier

    Args:
        dataset: the dataset to be classified
        training_data: data to be used to train the model
        label: name of label in training data set to be used for labeling

    Return:
         A Random Forest classifier
    '''
    print("Initializing Random Forest Classifier...")

    # Convert training data to labeled raster
    n_bands, rows, cols = dataset.bands_data.shape
    shapes = list(zip(training_data['geometry'], training_data[label]))
    labeled_pixels = rio.features.rasterize(shapes=shapes, out_shape=(rows, cols), fill=0, transform=dataset.transform)

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
    Runs a skitlearn Random Forest classifier on a given dataset

    Returns:
        Classified image
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

def load_satellite_imgs():
    '''
    Takes all satellite images in the data directory specified in config.py and creates a list of SatelliteIm objects.
    Images will be cropped by boundary shape file if one exists.

    Returns:
        List of SatelliteImg objects
    '''
    images = []
    fire_boundary = None

    # Get files from data directory specified in config.py
    try:
        files = os.listdir(cfg.data_dir)
    except FileNotFoundError:
        sys.exit("No files found in the filepath specified by config.py")

    # Iterate the files and find boundary shape file and all satellite images
    # Instantiate an SatelliteImg object for each image and load band data
    for f in files:
        if f.endswith('boundary.shp'):
            fire_boundary = os.path.join(cfg.data_dir, f)
        elif f.endswith('.img') or f.endswith('.tif'):
            # Find all post fire rasters. Should end with 8 digits defining the date
            match = re.search(r'\d{4}\d{2}\d{2}', f)
            if bool(match):
                date = datetime.datetime.strptime(match.group(), '%Y%m%d').date()
                img = SatelliteImg(os.path.join(cfg.data_dir, f), date)
                img.load_band_data(fire_boundary)
                images.append(img)

    return images




