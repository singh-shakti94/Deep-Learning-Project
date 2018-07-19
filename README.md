# A Ship or an Iceberg?
### Project Description
The aim of this project is to buid a classifier that can identify if a remotely sensed target is a Ship or a drifting Iceberg. This is an attempt to solve the problem stated in one of the competitions at Kaggle.com ([here](https://www.kaggle.com/c/statoil-iceberg-classifier-challenge)).
### Dataset
The dataset for this project is borrowed from its Kaggle competition page (link to dataset [here](https://www.kaggle.com/c/statoil-iceberg-classifier-challenge/data))

The data is provided in .json format (train.json and test.json). The files consist of a list of images and for each image the following fields
* **id**: id of the image
* **band_2, band_2**: flattened image data. each band list consist of 5627 elements corresponding to 75x75 pixel values. (true meaning of these values still need to be understood [[satellite imagery](https://www.kaggle.com/c/statoil-iceberg-classifier-challenge#Background)]).
* **inc_angle**: inclination angle at which the image was taken.
* **is_iceberg**: this field exists only in train.json which if set to 1 indicates that the image is an iceberg, and 0 if its is a ship.

### Milestones
In order to work for this project, Some milestones have been agreed upon to mark the progress of the project.
- **July 13th 2018** By this date we will be able to complete the background needed for this project and will be able to come up with a simple CNN network in python tensorflow.
- **July 25th 2018** By this date we will be able to finalize the simple CNN classifier that we will create, by finalize I mean applying some techniques to increase the accuracy. Moreover, we will be able to come up with a pre-trained classifier (possibly VGG network) trained on the training set for this problem.
- **August 2nd 2018** The finilaization of both the processes (Simple Classifier and pre-trained network) will be done and we will be ready to present our work.

### Work Log
* **july 6th 2018** **Shakti and Nikhil** working on the backgroud of Satellite Imagery
  * Synthetic-aperture Radar (SAR) [Wiwkipedia link](https://en.wikipedia.org/wiki/Synthetic-aperture_radar)
  * Deep Learning for Target Classification from SAR Imagery [link to paper](https://arxiv.org/pdf/1708.07920.pdf)
  
* **july 13th 2018** A simple convolution neural network is has been created. We are using 3 convolution layers and a fully connected layer to get predictions. the details for network are listed below :
  * Input : flattened data points (shape = batch_size x 5625) of 75 x 75 images 
  * Output : one-hot vector of predicted class (shape = batch_size x 2) 
  Please refer file CNN.py for the code.
  * Convolution layers :

    | Layer Index |   inputs    |   outputs   | filter shape | stride | pooling-stride | activation | 
    | ----------- |:-----------:| -----------:| ------------:| ------:| --------------:| ----------:|
    |      1      | -1x75x75x1  | -1x38x38x32 |     5x5      |   1    | max pooling - 2|   ReLU     |
    |      2      | -1x38x38x32 | -1x19x19x64 |     5x5      |   1    | max pooling - 2|   ReLU     |
    |      3      | -1x19x19x64 | -1x10x10x128|     5x5      |   1    | max pooling - 2|   ReLU     |
  * Fully connected layer :
    * number of nodes : 1024
    * input : 10 x 10 x 128 (reshaped to ? X 1024)
    * output size : batch_size x 1024
    * activation : ReLU
  * An attempt to apply dropout is being done (more work on it comming soon)
* **july 14th 2018** A simple 3D convolution neural network is been created. We are using 1 convolution layers and a fully connected layer to get predictions. the details for input and output for network are listed below :
  * Input : flattened data points (shape = batch_size x 16875) of 75 x 75 x 3 images 
  * Output : one-hot vector of predicted class (shape = batch_size x 2) 
  Please refer file CNN3D.py for the code.
* **july 19th 2018** Further detailed exploration of the dataset is done by visualizing random samples from the dataset for each class (file : data_exploration.ipynb). What we found out is that band_2 of most of the images given is comparatively noisy and apparently is not helping enough. We tried to create another cahnnel by combining the given two channels(sum or average of band_1 and band_2) and found out that even if we sum the two given bands, we get a channel that is less less noisy and can be used as a third channel in our convolution model. Although there wasn't any significant difference between the two combinations (sum and average), I personally found the summed version more helpful(we'll see how will that work out in the model).
  * The model that won the original competition isn't available publically, but other baseline models having fairly high accuracy are available publically. Most of them are implemented in keras (actually we didn't find any implemented in TensorFlow), so we are trying to get it to run as soon as possible.
  * Work is also in progress for our version of convolution network in TensorFlow. We will be modifying our 2d convolution implementation to work with three channels (band_1, band_2, band_1+band_2) and training stats will be shared here soon.
  * One more thing that I am working on is to add more images to the trainig dataset. I have gone through several data augmentation techniques that we can apply to availabel images in order to generate new images. these techniques include :
     * Rotation of image around its centre
     * zoomed crop of an image (since the objects are centered in all the images). may be take a 50x50 or 60x60 center crop of some images
  
  It would be helpful to increase the dataset to better train the network.
  * stats on training results will be updated soon.
### References
- Background on Satellite imaging https://www.kaggle.com/c/statoil-iceberg-classifier-challenge#Background
