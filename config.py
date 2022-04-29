# Friendly name for the wildfire to be analyzed
name = "Kårböle"

# UTM zone matching the satellite imagery
#utm_zone = 12 # Montana UTM zone
utm_zone = 33 # Kårböle UTM zone

# Spectral band mapping (depends on satellite program)
#bands = {'blue': 2, 'green': 3, 'red': 4, 'NIR': 5, 'SWIR': 6, 'SWIR2': 7} # Landsat8
bands = {'blue': 2, 'green': 3, 'red': 4, 'NIR': 9, 'SWIR': 11, 'SWIR2': 12} # Sentinel-2

# Path to directory of input data files
data_dir = 'data_files/karbole/'

# dBNR thresholds and corresponding colors and labels for plotting. Dimensions must match
bounds = [-0.5, 0.1, 0.27, 0.440, 0.660, 1.3]  # dNBR threshold values as defined by UN-SPIDER
labels = ['Unburned', 'Low Severity', 'Moderate-low Severity', 'Moderate-high Severity', 'High Severity']
colors = ['green', 'yellow', 'orange', 'red', 'purple'] #


