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

    # Spectral band mapping data structure for different satellites
    bands = cfg.bands

    def __init__(self, file_path, date):
        self.file_path = file_path
        self.date = date
        self.crs = ""
        self.transform = ""
        self.bands_data = []
        self.extent = []

    def load_band_data(self, fire_boundary=None):
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

    # TODO: Needed?
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

def dnbr(prefire, postfire):
    '''
    TODO: This is where you should write a docstring.
    '''
    return prefire.nbr() - postfire.nbr()

def dndvi(prefire, postfire):
    '''
    TODO: This is where you should write a docstring.
    '''
    return prefire.ndvi() - postfire.ndvi()

def plot_dnbr(dnbr, date, crs):
    '''
    TODO: This is where you should write a docstring.
    '''
    fig, ax = plt.subplots(1, 1, figsize=(10, 10), subplot_kw=dict(projection=crs))

    # Set colors for plotting and classes for dNBR
    cmap = matplotlib.colors.ListedColormap(cfg.colors)
    bounds = [-0.5, 0.1, 0.27, 0.440, 0.660, 1.3]  # dNBR threshold values as defined by UN-SPIDER
    norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)

    # Legend
    labels = cfg.labels
    handles = generate_handles(labels, cfg.colors)
    ax.legend(handles, labels, fontsize=10, loc='lower left', framealpha=1)

    ax.imshow(dnbr, cmap=cmap, norm=norm, transform=crs)
    ax.set_title("dNBR " + cfg.name + ", Date: " + str(date))

    # Save the dNBR map
    # Todo: Create result directory
    fig.savefig('output_maps/dnbr_' + str(date) + '.png', dpi=300, bbox_inches='tight')

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

def load_satellite_imgs():
    '''
    TODO: This is where you should write a docstring.
    '''
    images = []
    fire_boundary = None

    # Get files from data directory specified in config.py
    try:
        files = os.listdir(cfg.data_dir)
    except FileNotFoundError:
        sys.exit("No files found in the filepath specified by config.py")

    # Iterate the files and find boundary shape file and all satellite images
    # Instantiate an SatelliteImg object for each image and load band date
    for f in files:
        if f.endswith('boundary.shp'):
            fire_boundary = os.path.join(cfg.data_dir, f)
        elif f.endswith('.img'):
            # Find all post fire rasters. Should end with 8 digits defining the date
            match = re.search(r'\d{4}\d{2}\d{2}', f)
            if bool(match):
                date = datetime.datetime.strptime(match.group(), '%Y%m%d').date()
                img = SatelliteImg(os.path.join(cfg.data_dir, f), date)
                img.load_band_data(fire_boundary)
                images.append(img)

    return images





