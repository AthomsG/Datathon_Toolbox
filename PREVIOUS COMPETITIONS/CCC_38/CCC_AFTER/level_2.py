import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from utils import load_images, count_pixels

# REGRESSION PROBLEM - LEVEL 2

path_to_train_images = 'levels/level_02/train_data'
path_to_test_images  = 'levels/level_02/test_data'
path_to_train_class  = 'levels/level_02/train_data_labels.csv'

train_data   = load_images(path_to_train_images)
test_data    = load_images(path_to_test_images)

y_true = np.loadtxt(path_to_train_class, 
                    delimiter=',', 
                    skiprows=0, # don't skip header
                    dtype=int)

pixel_counts = list()
# count black pixels in each image
for image in tqdm(train_data, desc='Counting pixels'):
    pixel_counts.append(count_pixels(image, [0,0,0]))

# check if theres a linear relation between number of black pixels and number of animals
plt.plot(pixel_counts, y_true, 'o')
plt.xlabel('Number of black pixels')
plt.ylabel('Number of animals')
plt.show()

'''
we see that there's a clear linear relation between number of black pixels 
and number of animals. Each animal produces ~10 black pixels.
'''

# test accuracy on training data - sanity check
y_pred = np.round(np.array(pixel_counts)/10).astype(int)
print('Accuracy on training data: ', np.mean(y_true == y_pred))

test_pixel_counts = list()
for image_test in tqdm(test_data, desc='Counting pixels'):
    test_pixel_counts.append(count_pixels(image_test, [0,0,0]))

y_pred = np.round(np.array(test_pixel_counts)/10).astype(int)
np.savetxt('outputs/output_2.txt', y_pred, delimiter=',', fmt='%d')

from IPython import embed; embed()