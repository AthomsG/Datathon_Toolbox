import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from utils import load_images, count_pixels, clear_clouds, draw_circle

# COORDINATES PROBLEM WITH CLOUDS - LEVEL 4

path_to_train_images = 'levels/level_04/train_data'
path_to_test_images  = 'levels/level_04/test_data'
path_to_train_class  = 'levels/level_04/train_data_labels.csv'

train_data   = load_images(path_to_train_images)
test_data    = load_images(path_to_test_images)

y_true = np.loadtxt(path_to_train_class, 
                    delimiter=',', 
                    skiprows=0, # don't skip header
                    dtype=int)

# remove clouds from images
# train
cleared_train_data = list()
for i in tqdm(range(0, len(train_data)-1, 2), desc='Clearing clouds from training data'):
    cleared_train_data.append(clear_clouds(train_data[i], train_data[i+1]))
# test
cleared_test_data = list()
for i in tqdm(range(0, len(test_data)-1, 2), desc='Clearing clouds from test data'):
    cleared_test_data.append(clear_clouds(test_data[i], test_data[i+1]))

# store black pixels coords from train and test data
# train
pixel_coords_train = list()
for image in tqdm(cleared_train_data, desc='Counting pixels - train'):
    pixel_coords_train.append(count_pixels(image, [0,0,0], save_pixel_coords=True))
# test
pixel_coords_test = list()
for image in tqdm(cleared_test_data, desc='Counting pixels - test'):
    pixel_coords_test.append(count_pixels(image, [0,0,0], save_pixel_coords=True))

# compute the mean coordinates for each picture
mean_coords_train = []
for coords in pixel_coords_train:
    mean_coords_train.append(np.round(np.mean(coords, axis=0)))

# compute mean euclidean distance between predicted and true animal positions - sanity check
distances = list()
for i in tqdm(range(len(mean_coords_train)), desc='Computing distances'):
    distances.append(np.linalg.norm(mean_coords_train[i] - y_true[i]))

print('Mean Euclidean distance: ', np.mean(distances)) 

# draw circles centered on mean_coords_train - sanity check
for i in tqdm(range(len(cleared_train_data)), desc='Drawing circles'):
    img = draw_circle(cleared_train_data[i], mean_coords_train[i], 4, [255,0,0])
    plt.imsave('outputs/level_4_train_image_center/train_image_' + str(i) + '.png', img)

# compute the mean coordinates for each test picture
mean_coords_test = []
for coords in pixel_coords_test:
    mean_coords_test.append(np.round(np.mean(coords, axis=0)))

# test data - prediction
np.savetxt('outputs/output_4.txt', mean_coords_test, delimiter=',', fmt='%d')

from IPython import embed; embed()