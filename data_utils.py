
from tensorflow import keras
import os


def data_encode(X_train, y_train, X_test, y_test, k, num_classes):

    for i in range(k):
#         print(x_train[i].shape, ' ', y_train[i].shape)
#         print(x_test[i].shape, ' ', y_test[i].shape)


        # Convert class vectors to binary class matrices.
        y_train[i] = keras.utils.to_categorical(y_train[i], num_classes)
        y_test[i] = keras.utils.to_categorical(y_test[i], num_classes)


        X_train[i] = X_train[i].astype('float32')
        X_test[i] = X_test[i].astype('float32')

        X_train[i] /= 255  
        X_test[i] /= 255
    return X_train, y_train, X_test, y_test
