from random import choice
import cv2
import numpy as np
import keras.preprocessing.image as prep
import pandas as pd


data2 = pd.read_json('train.json') # this is a dataframe

X_band_1=np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in data2["band_1"]])
X_band_2=np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in data2["band_2"]])
train_data = np.concatenate([X_band_1[:, :, :, np.newaxis], X_band_2[:, :, :, np.newaxis],
                            X_band_1[:, :, :, np.newaxis]+X_band_2[:, :, :, np.newaxis]], axis=-1)

train_targets=np.array(data2["is_iceberg"]).reshape([-1,1])
total_data=map(lambda x,y:(x,y),train_data,train_targets)
final_images=[]
i=0
list1=[1,2,3]
list2=[2,3,4]


for image,target in total_data:

    img_v=cv2.flip(image,0)
    img_h=cv2.flip(image,1)
    img_rot=np.rot90(image, k=choice([0, 1, 2, 3]), axes=(0, 1))
    img_prep=prep.random_shift(image, wrg=0.1, hrg=0.1,
                                  row_axis=0, col_axis=1, channel_axis=2)
    final_images.append((img_v,target))
    final_images.append((img_h, target))
    final_images.append((img_rot, target))
    final_images.append((img_prep, target))
c=total_data+final_images