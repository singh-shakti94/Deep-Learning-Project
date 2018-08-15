import pandas as pd
import numpy as np
import random
import cv2 # Used to manipulated the images

from sklearn.cross_validation import train_test_split

np.random.seed(1337) # The seed I used - pick your own or comment out for a random seed. A constant seed allows for better comparisons though
import matplotlib as plt
import keras.preprocessing.image as prep
# # Import Keras
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten, Activation
from keras.layers.convolutional import Convolution2D as Conv2D, MaxPooling2D
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam

train = pd.read_json("train.json")
test = pd.read_json("test.json")
X_band_1=np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in train["band_1"]])
X_band_2=np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in train["band_2"]])
target_train=train['is_iceberg']
X_train = np.concatenate([X_band_1[:, :, :, np.newaxis], X_band_2[:, :, :, np.newaxis],((X_band_1+X_band_2)/2)[:, :, :, np.newaxis]], axis=-1)
X_train_cv, X_valid, y_train_cv, y_valid = train_test_split(X_train, target_train, random_state=1, train_size=0.75)
# image generator generating image tensors from the data
gen = prep.ImageDataGenerator(horizontal_flip=True,
                              vertical_flip=True,
                              width_shift_range=2,
                              height_shift_range=2,
                              channel_shift_range=0,
                              zoom_range=0.2,
                              rotation_range=10)

gen_op = gen.flow(x=X_train_cv, y=y_train_cv,batch_size=32, seed=10)
gen_val = gen.flow(x=X_valid, y=y_valid,batch_size=32, seed=10)
def getModel():
    #Building the model
    gmodel=Sequential()
    #Conv Layer 1
    gmodel.add(Conv2D(64, kernel_size=(3, 3),activation='relu', input_shape=(75, 75, 3)))
    # gmodel.add(BatchNormalization())
    gmodel.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    gmodel.add(Dropout(0.2))

    #Conv Layer 2
    gmodel.add(Conv2D(128, kernel_size=(3, 3), activation='relu' ))
    # gmodel.add(BatchNormalization())
    gmodel.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    gmodel.add(Dropout(0.2))


    #Conv Layer 3
    gmodel.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    # gmodel.add(BatchNormalization())
    gmodel.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    gmodel.add(Dropout(0.2))

    #Conv Layer 4
    gmodel.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    # gmodel.add(BatchNormalization())
    gmodel.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    gmodel.add(Dropout(0.2))


    #Flatten the data for upcoming dense layers
    gmodel.add(Flatten())

    #Dense Layers
    gmodel.add(Dense(512))
    gmodel.add(Activation('relu'))
    gmodel.add(Dropout(0.2))

    #Dense Layer 2
    gmodel.add(Dense(256))
    gmodel.add(Activation('relu'))
    gmodel.add(Dropout(0.2))

    #Sigmoid Layer
    gmodel.add(Dense(1))
    gmodel.add(Activation('sigmoid'))

    mypotim=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    gmodel.compile(loss='binary_crossentropy',
                  optimizer=mypotim,
                  metrics=['accuracy'])
    gmodel.summary()
    return gmodel

def get_callbacks(filepath):
    es = EarlyStopping('val_loss', patience=10, mode="min")
    msave = ModelCheckpoint(filepath, save_best_only=True)
    reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss',
                                       factor=0.1,
                                       patience=7,
                                       verbose=0,
                                       epsilon=1e-4,
                                       mode='min')

    tnb = TensorBoard(
        log_dir="logs/run_b",
        histogram_freq=2,
        write_graph=True,
        write_images=True
    )
    return [es, msave, reduce_lr_loss, tnb]
file_path = ".model_weights.hdf5"
callbacks = get_callbacks(filepath=file_path)



gmodel=getModel()
gmodel.fit_generator(generator=gen_op,
                    epochs=20,
                     callbacks=callbacks,
                    validation_data=(X_valid,y_valid))

gmodel.load_weights(filepath=file_path)
score = gmodel.evaluate(X_valid, y_valid, verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


X_band_test_1=np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in test["band_1"]])
X_band_test_2=np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in test["band_2"]])
X_test = np.concatenate([X_band_test_1[:, :, :, np.newaxis]
                          , X_band_test_2[:, :, :, np.newaxis]
                         , ((X_band_test_1+X_band_test_2)/2)[:, :, :, np.newaxis]], axis=-1)
predicted_test=gmodel.predict_classes(X_test)

submission = pd.DataFrame()
submission['id']=test['id']
submission['is_iceberg']=predicted_test.reshape((predicted_test.shape[0]))
submission.to_csv('subkeras.csv', index=False)
from keras.utils import plot_model
plot_model(gmodel, to_file='model1.png')
