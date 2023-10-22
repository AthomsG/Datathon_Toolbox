from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from keras.optimizers import Adam

def create_classifier(input_size, output):
    model = Sequential([
        Conv2D(32, kernel_size=3, activation='relu', input_shape=(input_size[0],input_size[1],3)),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, kernel_size=3, activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(128, kernel_size=3, activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(256, kernel_size=3, activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(256, activation='relu'),
        Dense(128, activation='relu'),
        Dense(output, activation='softmax')
    ])
    
    return model

