from dl_utils import load_images, resize_images
from models.regression_model import create_regression_model

from keras.optimizers import Adam
import numpy as np

SAVE_MODEL = False
LOAD_MODEL = False

# we need 80% accuracy!

path_to_train_images = '../levels/level_02/train_data'
path_to_test_images  = '../levels/level_02/test_data'
path_to_train_class  = '../levels/level_02/train_data_labels.csv'
# true labels
path_to_test_labels  = '../outputs/output_2.txt'; y_true = np.loadtxt(path_to_test_labels, delimiter=',', dtype=int)

X_train   = load_images(path_to_train_images)
y_train   = np.loadtxt(path_to_train_class, 
                    delimiter=',', 
                    skiprows=0, # don't skip header
                    dtype=int)

X_test   = load_images(path_to_test_images)

size=(200, 200) # input size of the model

X_train = resize_images(X_train, size)
X_test = resize_images(X_test, size)

model = create_regression_model(size, 1)

model.compile(optimizer=Adam(learning_rate=0.001), 
              loss='mean_squared_error', 
              metrics=['mse'])

if LOAD_MODEL: model.load_weights('model_weights/level_2_model.keras')
if not LOAD_MODEL:model.fit(X_train, 
                            y_train, 
                            epochs=10, 
                            batch_size=32, 
                            validation_split=0.2)

if SAVE_MODEL: model.save('model_weights/level_2_model.keras')

# predict test data
y_pred = model.predict(X_test)

# round values to the closest integer between 0 and 5
y_pred_rounded = np.round(y_pred)
y_pred_rounded = np.clip(y_pred_rounded, 0, 5)

# evaluate model root mean squared error
print('RMSE on test data: ', np.sqrt(np.mean((y_true - y_pred_rounded)**2)))
print('Accuracy on test data: ', 100*np.sum(y_true==y_pred_rounded.flatten().astype(int))/len(y_true))

from IPython import embed; embed()