import json
import numpy as np
import matplotlib.pyplot as plt

# load data
with open("train.json") as f:
    data = json.load(f)

# plot several examples
for i in range(2):
    b1 = np.array(data[i]["band_1"])
    b2 = np.array(data[i]["band_2"])
    print(data[i]["is_iceberg"])
    b1 = b1.reshape([75, 75])
    b2 = b2.reshape([75, 75])
    print(b1.shape)
    plt.imshow(b1)
    plt.show()
    plt.imshow(b2)
    plt.show()