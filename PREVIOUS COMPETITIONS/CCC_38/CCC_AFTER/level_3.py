import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from utils import load_images, count_pixels, clear_clouds

# REGRESSION PROBLEM WITH CLOUDS - LEVEL 3

path_to_train_images = 'levels/level_03/train_data'
path_to_test_images  = 'levels/level_03/test_data'
path_to_train_class  = 'levels/level_03/train_data_labels.csv'

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

# count black pixels in train and test data
# train
pixel_counts_train = list()
for image in tqdm(cleared_train_data, desc='Counting pixels - train'):
    pixel_counts_train.append(count_pixels(image, [0,0,0]))
# test
pixel_counts_test = list()
for image in tqdm(cleared_test_data, desc='Counting pixels - test'):
    pixel_counts_test.append(count_pixels(image, [0,0,0]))

# test accuracy on training data - sanity check
y_pred = np.round(np.array(pixel_counts_train)/10).astype(int)
print('Accuracy on training data: ', np.mean(y_true == y_pred))

# test data - prediction
y_pred = np.round(np.array(pixel_counts_test)/10).astype(int)
np.savetxt('outputs/output_3.txt', y_pred, delimiter=',', fmt='%d')

from IPython import embed; embed()