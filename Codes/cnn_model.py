

from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
import functools




def cnn_build(X_train, k, num_classes):

    model = []
    for i in range(k):
        modelt = Sequential()
        modelt.add(Conv2D(32, (3, 3), padding='same',
                         input_shape=X_train[i][0].shape))
        modelt.add(Conv2D(32, (3, 3)))
        modelt.add(Activation('relu'))
        modelt.add(MaxPooling2D(pool_size=(2, 2)))
        modelt.add(Dropout(0.25))

        modelt.add(Conv2D(64, (3, 3), padding='same'))
        modelt.add(Activation('relu'))
        modelt.add(Conv2D(64, (3, 3)))
        modelt.add(Activation('relu'))
        modelt.add(MaxPooling2D(pool_size=(2, 2)))
        modelt.add(Dropout(0.25))

        modelt.add(Flatten())
        modelt.add(Dense(512))
        modelt.add(Activation('relu'))
        modelt.add(Dropout(0.5))
        modelt.add(Dense(num_classes))
        modelt.add(Activation('softmax'))

        # initiate RMSprop optimizer
        opt = keras.optimizers.RMSprop(lr=0.0001, decay=1e-6)

        # Let's train the model using RMSprop
        modelt.compile(loss='categorical_crossentropy',
                      optimizer=opt,
                      metrics=['accuracy'])
        model.append(modelt)
    print('model ready')
    return model
    # =============================================================================

    
    
def cnn_train(model, X_train, y_train, X_test, y_test, k, batch_size, epochs, p):
    # =============================================================================
    history = []
    for i in range(k):
        historyt = model[i].fit(X_train[i], y_train[i],
                  batch_size=batch_size,
                  epochs=epochs,
                  validation_data=(X_test[i], y_test[i]),
                  shuffle=True, verbose=p)
        history.append(historyt)
    print('ok')
    return model,history
    # =============================================================================
