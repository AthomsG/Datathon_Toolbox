import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from utils import load_images, count_pixels, clear_clouds, draw_circle, get_cluster_coords

PERFORM_SANITY_CHECK = False

# COORDINATES PROBLEM WITH CLOUDS - LEVEL 5

path_to_test_images  = 'levels/level_05/test_data'
test_data    = load_images(path_to_test_images)

# remove clouds from images
cleared_test_data = list()
for i in tqdm(range(0, len(test_data)-1, 2), desc='Clearing clouds from test data'):
    cleared_test_data.append(clear_clouds(test_data[i], test_data[i+1]))

# store black pixels coords from data
pixel_coords_test = list()
for image in tqdm(cleared_test_data, desc='Counting pixels - test'):
    pixel_coords_test.append(count_pixels(image, [0,0,0], save_pixel_coords=True))

# get cluster coordinates for all images
cluster_coords_test = list()
for coords in tqdm(pixel_coords_test, desc='Getting cluster coordinates'):
    cluster_coords_test.append(get_cluster_coords(coords))

if PERFORM_SANITY_CHECK: # saves plots with target circles slow to run... 
    # draw circles centered on cluster_coords_test - sanity check
    for i in tqdm(range(len(cleared_test_data)), desc='Drawing circles'):
        img = cleared_test_data[i]
        for animal_index in range(len(cluster_coords_test[i])):
            img = draw_circle(img, cluster_coords_test[i][animal_index], 4, [255,0,0])
        plt.imsave('outputs/level_5_train_image_center/train_image_' + str(i) + '.png', img)

# save cluster_coords_test to file
# flatten cluster_coords_test to a 2D array
flat_coords = np.reshape(cluster_coords_test, (len(cluster_coords_test), -1))
# save flat_coords to file
np.savetxt('outputs/cluster_coords_test.txt', flat_coords, delimiter=',', fmt='%d')

from IPython import embed; embed()