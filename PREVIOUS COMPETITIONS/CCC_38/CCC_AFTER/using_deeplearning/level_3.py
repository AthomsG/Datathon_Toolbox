from dl_utils import load_images, resize_images, clear_clouds
from models.regression_model import create_regression_model

from keras.optimizers import Adam
import numpy as np
from tqdm import tqdm

SAVE_MODEL = False
LOAD_MODEL = True

# same as level_2 but we need to clear the clouds!

path_to_train_images = '../levels/level_03/train_data'
path_to_test_images  = '../levels/level_03/test_data'
path_to_train_class  = '../levels/level_03/train_data_labels.csv'
# true labels
path_to_test_labels  = '../outputs/output_3.txt'; y_true = np.loadtxt(path_to_test_labels, delimiter=',', dtype=int)

X_train   = load_images(path_to_train_images)
y_train   = np.loadtxt(path_to_train_class, 
                    delimiter=',', 
                    skiprows=0, # don't skip header
                    dtype=int)

X_test   = load_images(path_to_test_images)

size=(200, 200) # input size of the model

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

model = create_regression_model(size, 1)

model.compile(optimizer=Adam(learning_rate=0.001), 
              loss='mean_squared_error', 
              metrics=['mse'])

if LOAD_MODEL: model.load_weights('model_weights/level_2_model.keras')
if not LOAD_MODEL:model.fit(cleared_X_train, 
                            y_train, 
                            epochs=10, 
                            batch_size=32, 
                            validation_split=0.2)

if SAVE_MODEL: model.save('model_weights/level_2_model.keras')

# predict test data
y_pred = model.predict(cleared_X_test)

# round values to the closest integer between 0 and 5
y_pred_rounded = np.round(y_pred)
y_pred_rounded = np.clip(y_pred_rounded, 0, 5)

# evaluate model root mean squared error
print('RMSE on test data: ', np.sqrt(np.mean((y_true - y_pred_rounded)**2)))
print('Accuracy on test data: ', 100*np.sum(y_true==y_pred_rounded.flatten().astype(int))/len(y_true))

from IPython import embed; embed()