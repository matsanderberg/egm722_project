from satelliteimg import *

crs = ccrs.UTM(cfg.utm_zone)

# Load all satellite images (as objects of class SatelliteImg) available for analysis into a list
images = load_satellite_imgs()

if (images):
    # Sort list of image objects by date. We want to make sure the pre fire raster is at index 0
    images.sort(key=lambda img: img.date)

    # Pre fire raster should be the one with the earliest date
    pre_fire = images[0]

    plot = {"bounds": cfg.bounds,
            "labels": cfg.labels,
            "colors": cfg.colors}
    # Calculate the dNBR and dNDVI for all available post fire images and plot
    for post_fire in images[1:]:
        dnbr = dnbr(pre_fire, post_fire)
        plot_burn_severity("dNBR", dnbr, post_fire.date, plot, crs)
        dndvi = dndvi(pre_fire, post_fire)
        plot_burn_severity("dNDVI", dndvi, post_fire.date, plot, crs)
        dndmi = dndmi(pre_fire, post_fire)
        plot_burn_severity("dNDMI", dndmi, post_fire.date, plot, crs)
else:
    print("No valid raster images found. Check your data directory.")

# Plot spectral indices for each image
for img in images:
    img.plot("NBR", crs)
    img.plot("NDVI", crs)
    img.plot("NDMI", crs)

# Supervised Learning with Random Forest
# Based on tutorial from: https://adaneon.com/image-analysis-tutorials/pages/part_four.html
# ----------------------------------------------------------------------------------------

# Load data
dataset = images[1]
training_data = gpd.read_file('data_files/Karbole/training_data.shp').to_crs(dataset.crs)
extent = dataset.get_extent()

# Initialize and run the classifier
classifier = init_random_forest(dataset, training_data, 'Classvalue')
classification = random_forest(classifier, dataset)

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
print("Classification with Random Forest classification")
print("Burned area (km2): " + str(burned_size))
print("Unburned area (km2): " + str(unburned_size))