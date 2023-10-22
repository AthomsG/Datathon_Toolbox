from dl_utils import load_images, resize_images
from models.classifier_model import create_classifier

from keras.optimizers import Adam
import numpy as np
from keras.utils import to_categorical

# we need 80% accuracy!

path_to_train_images = '../levels/level_01/train_data'
path_to_test_images  = '../levels/level_01/test_data'
path_to_train_class  = '../levels/level_01/train_data_labels.csv'
# true labels
path_to_test_labels  = '../outputs/output_1.txt'; y_true = np.loadtxt(path_to_test_labels, delimiter=',', dtype=int)

X_train   = load_images(path_to_train_images)
y_train   = np.loadtxt(path_to_train_class, 
                    delimiter=',', 
                    skiprows=0, # don't skip header
                    dtype=int)

X_test   = load_images(path_to_test_images)

size=(200, 200) # input size of the model

X_train = resize_images(X_train, size)
X_test = resize_images(X_test, size)

model = create_classifier(size, 2)

model.compile(optimizer=Adam(learning_rate=0.001), 
                loss='categorical_crossentropy', 
                metrics=['accuracy'])

model.fit(X_train, 
          to_categorical(y_train), 
          epochs=10, 
          batch_size=32, 
          validation_split=0.2)

model.save('model_weights/level_1_model.keras')

# predict test data
y_pred = model.predict(X_test)
# evaluate model
print('Accuracy on test data: ', np.mean(y_true == np.argmax(y_pred, axis=1)))
# 97.4% accuracy

from IPython import embed; embed()