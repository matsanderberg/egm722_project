# TODO: Header
#
#
#
#

from dnbr import *

crs = ccrs.UTM(cfg.utm_zone)  # TODO: this should match with the CRS of our image. Can ccrs.epsg be used?

# Load all satellite images (as objects of class SatelliteImg) available for analysis into a list
images = load_satellite_imgs()

if (images):
    # Sort list of image objects by data. We want to make sure the pre fire raster is at index 0
    images.sort(key=lambda img: img.date)

    # Pre fire raster should be the one with the earliest date
    pre_fire = images[0]
    # Calculate the dNBR for all available post fire images and plot
    for post_fire in images[1:]:
        dnbr = dnbr(pre_fire, post_fire)
        plot_dnbr(dnbr, post_fire.date, crs)

    # Todo: calculate statistics
    # Todo: save to datebase?
else:
    print("No valid raster images found. Check your data directory.")

# This examples calculates NDVI and NDMI indices pre and for a given post fire image
# and plots the on a 2x2 axes
# --------------------------------------------------------------------------------------- #
pre_ndvi = images[0].ndvi()
post_ndvi = images[1].ndvi()
#dndvi = dndvi(pre_fire, post_fire)

pre_ndmi = images[0].ndmi()
post_ndmi = images[1].ndmi()

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(24, 16), subplot_kw=dict(projection=crs))

h = ax1.imshow(pre_ndmi, cmap='PuBuGn')
fig.colorbar(h, ax=ax1)
ax1.set_title("Pre Fire NDMI, " + str(images[0].date))
h = ax2.imshow(post_ndmi, cmap='PuBuGn')
fig.colorbar(h, ax=ax2)
ax2.set_title("Post Fire NDMI, " + str(images[1].date))
h = ax3.imshow(pre_ndvi, cmap='RdYlGn')
fig.colorbar(h, ax=ax3)
ax3.set_title("Pre Fire NDVI, " + str(images[0].date))
h = ax4.imshow(post_ndvi, cmap='RdYlGn')
fig.colorbar(h, ax=ax4)
ax4.set_title("Post Fire NDVI, " + str(images[1].date))

# Save the figure
fig.savefig('output_maps/ndmi_ndvi.png', dpi=300, bbox_inches='tight')


# Another (more elaborate) way of creating the color bar
# divider = make_axes_locatable(ax)
# cax = divider.append_axes("right", size="5%", pad=0.1, axes_class=plt.Axes)
# cbar = fig.colorbar(h, ax=ax, fraction=0.035, pad=0.04, ticks=[-0.2, 0.18, 0.35, 0.53, 1])
# cbar.ax.set_yticklabels(['Unburned', 'Low Severity', 'Moderate-low Severity', 'Moderate-high Severity', 'High Severity'])


# Supervised Learning with Random Forest
# Based on tutorial from: https://adaneon.com/image-analysis-tutorials/pages/part_four.html
# ----------------------------------------------------------------------------------------

# Load data
dataset = SatelliteImg('data_files/karbole_sentinel2_20180802.img', '20180802')
dataset.load_band_data('data_files/fire_boundary.shp')
training_data = gpd.read_file('data_files/training_data.shp').to_crs(dataset.crs)
extent = dataset.get_extent()

# Initialize and run the classifier
classifier = init_random_forest(dataset, training_data, 'Classvalue')
classification = random_forest(classifier, dataset)

#print("The three classes are: " + str(classes))
#print("Total number of training labels: " + str(training_labels.size))
#print("Total number of training sample size: " + str(training_samples.size))

# Plot classification image together with an Infrared band combination for comparison
fig, (ax1, ax2) = plt.subplots(1, 2, figsize =(18, 15), subplot_kw=dict(projection=crs))
ax1.set_title("Infrared band combination")
img_display(dataset.bands_data, ax1, [7, 3, 2], crs, extent)
ax2.set_title("Classification with Random Forest")
labels = ['Burned', 'Unburned', 'Water']
colors = ['red', 'green', 'blue']
handles = generate_handles(labels, colors)
cmap = matplotlib.colors.ListedColormap(colors)
ax2.legend(handles, labels, fontsize=10, loc='lower left', framealpha=1)
ax2.imshow(classification, cmap=cmap, transform=crs, extent=extent)

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

#reclass_dnbr = reclassify_dbnr(dnbr, 0.1)
#burned = reclass_dnbr[reclass_dnbr == 1]
#burned_size = (burned.size*20*20)/1000000
#unburned = reclass_dnbr[reclass_dnbr == 2]
#unburned_size = (unburned.size*20*20)/1000000
#print("With dNBR")
#print("Burned area (km2): " + str(burned_size))
#print("Unburned area (km2): " + str(unburned_size))
