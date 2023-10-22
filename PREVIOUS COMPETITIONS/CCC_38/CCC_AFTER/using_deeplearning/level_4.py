from dl_utils import load_images, resize_images, clear_clouds, draw_circle
from models.regression_model import create_regression_model

from keras.optimizers import Adam
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

SAVE_MODEL = True
LOAD_MODEL = False

size=(200, 200) # input size of the model

path_to_train_images = '../levels/level_04/train_data'
path_to_test_images  = '../levels/level_04/test_data'
path_to_train_class  = '../levels/level_04/train_data_labels.csv'
# true labels
path_to_test_labels  = '../outputs/output_4.txt'; y_true = np.loadtxt(path_to_test_labels, delimiter=',', dtype=int)

X_train   = load_images(path_to_train_images)
y_train   = np.loadtxt(path_to_train_class, 
                    delimiter=',', 
                    skiprows=0, # don't skip header
                    dtype=int)

X_test   = load_images(path_to_test_images)

X_train = resize_images(X_train, size)
X_test = resize_images(X_test, size)

# clear clouds from images
# train
cleared_X_train = list()
for i in tqdm(range(0, len(X_train)-1, 2), desc='Clearing clouds from training data'):
    cleared_X_train.append(clear_clouds(X_train[i], X_train[i+1]))
cleared_X_train = np.array(cleared_X_train)
# test
cleared_X_test = list()
for i in tqdm(range(0, len(X_test)-1, 2), desc='Clearing clouds from test data'):
    cleared_X_test.append(clear_clouds(X_test[i], X_test[i+1]))
cleared_X_test = np.array(cleared_X_test)

model = create_regression_model(size, 2)

model.compile(optimizer=Adam(learning_rate=0.001), 
              loss='mean_squared_error', 
              metrics=['mse'])

if LOAD_MODEL: model.load_weights('model_weights/level_4_model.keras')
if not LOAD_MODEL:model.fit(cleared_X_train, 
                            y_train, 
                            epochs=10, 
                            batch_size=32, 
                            validation_split=0.05)

if SAVE_MODEL: model.save('model_weights/level_4_model.keras')

# predict test data
y_pred = model.predict(cleared_X_test)

# round values to the closest integer between 0 and 5
y_pred_rounded = np.round(y_pred).astype(int)

# compute euclidean distance
def dist(v1, v2, single_point=False):
    return np.sqrt(np.sum((v2-v1)**2, axis=int(not single_point)))

# evaluate model euclidean distance
distances = dist(y_pred_rounded, y_true)
print('Mean Euclidean distance: ', np.mean(distances))

# draw circles centered on mean_coords_train - sanity check
for i in tqdm(range(len(cleared_X_test)), desc='Drawing circles'):
    img = draw_circle(img=cleared_X_test[i], 
                      center=y_pred_rounded[i], 
                      radius=4,
                      color=[255,0,0])
    plt.imsave('outputs/level_4_train_image_center/train_image_' + str(i) + '.png', img)

from IPython import embed; embed()