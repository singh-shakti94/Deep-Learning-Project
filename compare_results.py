import pandas as pd

MLP = pd.read_csv("submlp.csv")
keras_Ker = pd.read_csv("subkeras.csv")
resnet = pd.read_csv("subkerasResnet.csv")

import numpy as np
from sklearn.metrics import accuracy_score

list=[]

z1=accuracy_score(keras_Ker['is_iceberg'],resnet['is_iceberg'])
z2=accuracy_score(keras_Ker['is_iceberg'],MLP['is_iceberg'])

list.append(z1)

list.append(z2)

print list