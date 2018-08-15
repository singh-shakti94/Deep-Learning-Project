import numpy as np
from keras import layers
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, History, TensorBoard
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D
from keras.models import Model, load_model
from keras.preprocessing import image
from keras.utils import layer_utils, np_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
import pydot
import keras.preprocessing.image as prep
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
from keras.initializers import glorot_uniform
import scipy.misc
from matplotlib.pyplot import imshow
import pandas as pd

import keras.backend as K
K.set_image_data_format('channels_last')
K.set_learning_phase(1)


data = pd.read_json("train.json")
test=pd.read_json("test.json")

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
Y_train = np.concatenate(np.array_split(targets, 10, axis=0)[0:8], axis=0)
Y_test = np.concatenate(np.array_split(targets, 10, axis=0)[8:10], axis=0)

# to one-hot vectors
y_train = np_utils.to_categorical(Y_train, num_classes=2)
y_test = np_utils.to_categorical(Y_test, num_classes=2)

gen = prep.ImageDataGenerator(horizontal_flip=True,
                         vertical_flip=True,
                         width_shift_range=2,
                         height_shift_range=2,
                         channel_shift_range=0,
                         zoom_range=0.2,
                         rotation_range=10)

gen_op = gen.flow(x=X_train, y=y_train,batch_size=32, seed=10)
gen_val = gen.flow(x=X_test, y=y_test,batch_size=32, seed=10)

def identity_block(X, f, filters, stage, block):
    """
    Implementation of the identity block as defined in Figure 3

    Arguments:
    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    f -- integer, specifying the shape of the middle CONV's window for the main path
    filters -- python list of integers, defining the number of filters in the CONV layers of the main path
    stage -- integer, used to name the layers, depending on their position in the network
    block -- string/character, used to name the layers, depending on their position in the network

    Returns:
    X -- output of the identity block, tensor of shape (n_H, n_W, n_C)
    """

    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    # Retrieve Filters
    F1, F2 = filters

    # Save the input value. You'll need this later to add back to the main path.
    X_shortcut = X

    # First component of main path
    X = Conv2D(filters=F1, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name_base + '2a', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2a')(X)
    X = Activation('relu')(X)

    ### START CODE HERE ###


    X = Conv2D(filters=F2, kernel_size=(f, f), strides=(1, 1), padding='same', name=conv_name_base + '2b', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2b')(X)


    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)

    ### END CODE HERE ###

    return X


def convolutional_block(X, f, filters, stage, block, s=2):
    """
    Implementation of the convolutional block as defined in Figure 4

    Arguments:
    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    f -- integer, specifying the shape of the middle CONV's window for the main path
    filters -- python list of integers, defining the number of filters in the CONV layers of the main path
    stage -- integer, used to name the layers, depending on their position in the network
    block -- string/character, used to name the layers, depending on their position in the network
    s -- Integer, specifying the stride to be used

    Returns:
    X -- output of the convolutional block, tensor of shape (n_H, n_W, n_C)
    """

    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    # Retrieve Filters
    F1, F2 = filters

    # Save the input value
    X_shortcut = X

    ##### MAIN PATH #####
    # First component of main path
    X = Conv2D(filters=F1, kernel_size=(1, 1), strides=(s, s), padding='valid', name=conv_name_base + '2a', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2a')(X)
    X = Activation('relu')(X)

    X = Conv2D(filters=F2, kernel_size=(f, f), strides=(1, 1), padding='same', name=conv_name_base + '2b', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2b')(X)

    X_shortcut = Conv2D(filters=F2, kernel_size=(1, 1), strides=(s, s), padding='valid', name=conv_name_base + '1', kernel_initializer=glorot_uniform(seed=0))(X_shortcut)
    X_shortcut = BatchNormalization(axis=3, name=bn_name_base + '1')(X_shortcut)


    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)

    ### END CODE HERE ###
    return X

def ResNet50(input_shape, classes):
    """
    Implementation of the popular ResNet50 the following architecture:
    CONV2D -> BATCHNORM -> RELU -> MAXPOOL -> CONVBLOCK -> IDBLOCK*2 -> CONVBLOCK -> IDBLOCK*3
    -> CONVBLOCK -> IDBLOCK*5 -> CONVBLOCK -> IDBLOCK*2 -> AVGPOOL -> TOPLAYER

    Arguments:
    input_shape -- shape of the images of the dataset
    classes -- integer, number of classes

    Returns:
    model -- a Model() instance in Keras
    """

    # Define the input as a tensor with shape input_shape
    X_input = Input(input_shape)

    # Zero-Padding
    X = ZeroPadding2D((3, 3))(X_input)

    # Stage 1
    X = Conv2D(64, (7, 7), strides=(2, 2), name='conv1', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name='bn_conv1')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((3, 3), strides=(2, 2))(X)

    # Stage 2
    X = convolutional_block(X, f=3, filters=[128, 256], stage=2, block='a', s=1)
    X = identity_block(X, 3, [128, 256], stage=2, block='b')

    X = convolutional_block(X, f=3, filters=[64, 128], stage=3, block='a', s=2)
    X = identity_block(X, 3, [64, 128], stage=3, block='b')

    X = convolutional_block(X, f=3, filters=[256, 512], stage=4, block='a', s=2)
    X = identity_block(X, 3, [256, 512], stage=4, block='b')


    X = AveragePooling2D(pool_size=(2, 2), padding='same')(X)


    # output layer
    X = Flatten()(X)

    X=Dense(256,activation='relu')(X)
    X = Dense(classes, activation='softmax', name='fc' + str(classes), kernel_initializer=glorot_uniform(seed=0))(X)

    # Create model
    model = Model(inputs=X_input, outputs=X, name='ResNet50')
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()

    return model

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
        log_dir="logs/run_a",
        histogram_freq=2,
        write_graph=True,
        write_images=True
    )
    return [es, msave,reduce_lr_loss,tnb]

file_path = ".Resnetmodel_weights.hdf5"
callbacks = get_callbacks(filepath=file_path)

gmodel = ResNet50(input_shape=(75, 75, 3), classes=2)

gmodel.fit_generator(generator=gen_op,
                    epochs=20,
                    validation_data=(X_test,y_test),
                    callbacks=callbacks)

gmodel.load_weights(filepath=file_path)
score = gmodel.evaluate(X_test, y_test, verbose=1)


print('Test loss:', score[0])
print('Test accuracy:', score[1])

X_band_test_1=np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in test["band_1"]])
X_band_test_2=np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in test["band_2"]])
X_test = np.concatenate([X_band_test_1[:, :, :, np.newaxis]
                          , X_band_test_2[:, :, :, np.newaxis]
                         , ((X_band_test_1+X_band_test_2)/2)[:, :, :, np.newaxis]], axis=-1)
predicted_test=gmodel.predict(X_test)
y_pred = np.argmax(predicted_test, axis=1)

submission = pd.DataFrame()
submission['id']=test['id']
submission['is_iceberg']=y_pred.reshape((y_pred.shape[0]))
submission.to_csv('subkerasResnet.csv', index=False)


plot_model(gmodel, to_file='model.png')
