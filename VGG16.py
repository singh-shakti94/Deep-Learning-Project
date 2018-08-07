from keras.applications.vgg16 import VGG16
from keras.utils.vis_utils import plot_model

model = VGG16()
print(model.summary())
plot_model(model, to_file="vgg16.png")