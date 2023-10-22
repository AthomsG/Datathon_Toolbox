import numpy as np
from tqdm import tqdm

from utils import load_images, count_pixels

# ANIMAL CLASSIFICATION PROBLEM - LEVEL 1

path_to_train_images = 'levels/level_01/train_data'
path_to_test_images  = 'levels/level_01/test_data'
path_to_train_class  = 'levels/level_01/train_data_labels.csv'

train_data   = load_images(path_to_train_images)
test_data    = load_images(path_to_test_images)

y_true = np.loadtxt(path_to_train_class, 
                    delimiter=',', 
                    skiprows=0, # don't skip header
                    dtype=int)

# get coordinates of black pixels in each image
pixel_counts = list()
for image in tqdm(train_data, desc='Searching pixels'):
    pixel_counts.append(count_pixels(image, [0,0,0]))

# HYPOTHESIS - if there are no black pixels then image has no animals
y_pred = np.array([count != 0 for count in pixel_counts]).astype(int)
print('Accuracy: ', np.mean(y_true == y_pred))

# TEST DATA - Classification
test_pixel_counts = list()
for image in tqdm(test_data, desc='Searching pixels - Test Data'):
  test_pixel_counts.append(count_pixels(image, [0,0,0]))

y_pred = np.array([count != 0 for count in test_pixel_counts]).astype(int)
np.savetxt('outputs/output_1.txt', y_pred, delimiter=',', fmt='%d')

from IPython import embed; embed()