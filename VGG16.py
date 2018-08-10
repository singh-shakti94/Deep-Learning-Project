import numpy as np
import pandas as pd
from keras.applications.vgg16 import VGG16
from keras.utils.vis_utils import plot_model
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras.models import Model
from keras.optimizers import SGD
from keras.utils import np_utils
from sklearn.metrics import classification_report

# Import the data
data = pd.read_json("train.json")

# training data (for now using only band_1 for convolution)
# labels are in "is_iceberg" column where 0 value indicates a ship while 1 indicates iceberg
X_band_1 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in data["band_1"]])
X_band_2 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in data["band_2"]])
channel_3 = X_band_1 + X_band_2
new_data = np.concatenate([X_band_1[:, :, :, np.newaxis],
                             X_band_2[:, :, :, np.newaxis],
                             channel_3[:, :, :, np.newaxis]], axis=-1)

targets = data["is_iceberg"]

# split in test and train
split = np.array_split(new_data, 10, axis=0)
X_train = np.concatenate(split[0:8], axis=0)
X_test = np.concatenate(split[8:10], axis=0)
y_train = np.concatenate(np.array_split(targets, 10, axis=0)[0:8], axis=0)
y_test = np.concatenate(np.array_split(targets, 10, axis=0)[8:10], axis=0)


# to one-hot vectors
y_train = np_utils.to_categorical(y_train, num_classes=2)
y_test = np_utils.to_categorical(y_test, num_classes=2)

# image generator generating image tensors from the data
gen = ImageDataGenerator(horizontal_flip=True,
                         vertical_flip=True,
                         width_shift_range=2,
                         height_shift_range=2,
                         channel_shift_range=0,
                         zoom_range=0.2,
                         rotation_range=10)

gen_op = gen.flow(x=X_train, y=y_train, batch_size=10, seed=10)
gen_val = gen.flow(x=X_test, y=y_test, batch_size=10, seed=10)

def getModel():
    base_model = VGG16(weights='imagenet', include_top=False,
                       input_shape=X_train.shape[1:], classes=2)
    x = base_model.get_layer("block5_pool").output
    x = GlobalAveragePooling2D()(x)

    # add a fully conne3cted layer
    x = Dense(512, activation='relu', name="dense1")(x)
    x = Dropout(rate=0.2)(x)
    x = Dense(512, activation="relu", name="dense2")(x)
    x = Dropout(rate=0.3)(x)
    predictions = Dense(2, activation='softmax', name="output")(x)

    # this is the model we will train
    model = Model(inputs=base_model.input, outputs=predictions)

    # first: train only the top layers (which were randomly initialized)
    # i.e. freeze all convolutional VGG16 layers
    for layer in base_model.layers:
        layer.trainable = False

    sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='binary_crossentropy',
                  optimizer=sgd,
                  metrics=['accuracy'])

    return model

model = getModel()

# fit the data on model
model.fit_generator(generator=gen_op,
                    steps_per_epoch=40,
                    epochs=2,
                    validation_data=gen_val,
                    validation_steps=2)

predictions = model.predict(X_test, batch_size=10)
y_pred = np.argmax(predictions, axis=1)
classification_report(y_test, y_pred)
