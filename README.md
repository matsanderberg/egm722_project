# EGM722 Project - Installation instructions and how to use the code

## 1. Installation and setup

### 1.1 Prerequisties

To use the code provided in this repository, you'll first need to install `git` and `conda` on your computer. You can follow the
for installing git from [here](https://git-scm.com/downloads), and Anaconda from [here](https://docs.anaconda.com/anaconda/install/). 

Is Git needed (i.e. should it be possible to update the repository)?

### 1.2 Download/clone repository

Once you have these installed, __clone__ this repository to your computer by doing one of the following things:

1. Open GitHub Desktop and select __File__ > __Clone Repository__. Select the __URL__ tab, then enter the URL for this 
   repository.
2. Open __Git Bash__ (from the __Start__ menu), then navigate to your folder for this module. Execute the following command: 
   `git clone https://github.com/matsanderberg/egm722_project.git` to set up the repository. 
3. Click the green "clone or download" button above, and select "download ZIP" at the bottom of the menu. Once it's downloaded, unzip the file.

### 1.3 Create a conda environment

Once you have successfully cloned the repository, you can then create a `conda` environment to use the code.

To do this, use the environment.yml file provided in the repository. If you have Anaconda Navigator installed,
you can do this by selecting __Import__ from the bottom of the __Environments__ panel. 

Otherwise, you can open a command prompt (on Windows, you may need to select an Anaconda command prompt). Navigate
to the folder where you cloned this repository and run the following command:

```
C:\Users\username> conda env create -f environment.yml
```

## 2 Usage

### 2.1 config.py

The script in will use configurations provided in a python file that should be named `config.py`. This files should be created and stored in the same folder as `satelliteimg.py`. The config file defines band numbers (which will depend on satellite), where your datafiles are stored and definitions for map layout (coloring, labels etc). An example is provided below:

```
# Friendly name for the wild fire to be analyzed
name = "Kårböle"

# UTM zone matching the satellite imagery
utm_zone = 33

# Spectral band mapping (depends on satellite program)
bands = {'blue': 2, 'green': 3, 'red': 4, 'NIR': 9, 'SWIR': 11, 'SWIR2': 12} # Sentinel-2

# Path to directory of input data files
data_dir = 'data_files/'

# dBNR thresholds and corresponding colors and labels for plotting. Dimensions must match
bounds = [-0.5, 0.1, 0.27, 0.440, 0.660, 1.3]  # dNBR threshold values as defined by UN-SPIDER
labels = ['Unburned', 'Low Severity', 'Moderate-low Severity', 'Moderate-high Severity', 'High Severity']
colors = ['green', 'yellow', 'orange', 'red', 'purple']
```

### 2.2 Data files

The python script assumes data files are named a certain way. Any satellite image should be either a .tif or .img file and the file name must end with a date on the format YYYYMMDD, e.g. karbole_sentinel2_20180626.img. The script will assume the earliest image to be pre fire and the rest as post fire. At least one pre fire and one post fire should be provided. If the boundary file for the fire is provided the script will first crop images to the extent of the boundary. The boundary file should be in the format of a shape file and end with boundary.shp, e.g. fire_boundary.shp.

### 2.3 Run analysis

Provided below is an example how to use the satelliteimg.py to run a dNBR analysis on an arbirtray number of input files (the same code is also available in the repository in the egm_722.py).

```
from satelliteimg import *

crs = ccrs.UTM(cfg.utm_zone)

# Load all satellite images (as objects of class SatelliteImg) available for analysis into a list
images = load_satellite_imgs()

if (images):
    # Sort list of image objects by date. We want to make sure the pre fire raster is at index 0
    images.sort(key=lambda img: img.date)

    # Pre fire raster should be the one with the earliest date
    pre_fire = images[0]
    # Calculate the dNBR for all available post fire images and plot
    for post_fire in images[1:]:
        dnbr = dnbr(pre_fire, post_fire)
        plot_dnbr(dnbr, post_fire.date, crs)
else:
    print("No valid raster images found. Check your data directory.")
```

### 2.4 Output

Results from dNBR anaysis using the plot_dnbr function will be stored in a folder Result in the root folder for the script. If it doesn't exist it will be created. 

