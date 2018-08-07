from random import choice
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json


# load training dataframe
data2 = pd.read_json('train.json')

X_band_1=np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in data2["band_1"]])
X_band_2=np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in data2["band_2"]])
channel_3 = X_band_1 + X_band_2
train_data = np.concatenate([X_band_1[:, :, :, np.newaxis],
                             X_band_2[:, :, :, np.newaxis],
                             channel_3[:, :, :, np.newaxis]], axis=-1)

train_targets = np.array(data2["is_iceberg"])
init_data = list(map(lambda x, y: [x, y], train_data, train_targets))
aug_images = []

for image, target in init_data:
    img_v = np.flip(image, 0)
    img_h = np.flip(image, 1)
    img_rot = np.rot90(image, k=choice([1, 3]), axes=(0, 1))
    img_prep = np.roll(image, shift=5, axis=(0, 1))
    aug_images.append((img_v, target))
    aug_images.append((img_h, target))
    aug_images.append((img_rot, target))
    aug_images.append((img_prep, target))

final_data = np.array(init_data + aug_images)

plt.imshow(init_data[1][0][:, :, 0])
plt.imshow(aug_images[4][0][:, :, 0])
plt.imshow(aug_images[5][0][:, :, 0])
plt.imshow(aug_images[6][0][:, :, 0])
plt.imshow(aug_images[7][0][:, :, 0])
plt.show()

# save data as file
np.save("aug_data", final_data)
