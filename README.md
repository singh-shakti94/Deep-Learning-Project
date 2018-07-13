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
* **july 6th 2018** **Shakti** working on the backgroud of Satellite Imagery
  * Synthetic-aperture Radar (SAR) [Wiwkipedia link](https://en.wikipedia.org/wiki/Synthetic-aperture_radar)
  * Deep Learning for Target Classification from SAR Imagery [link to paper](https://arxiv.org/pdf/1708.07920.pdf)
* **july 13th 2018** A simple convolution neural network is has been created. We are using 3 convolution layers and a fully connected layer to get predictions. the details for network are listed below :
  * Input : flattened data points (shape = batch_size x 5625) of 75 x 75 images 
  * Output : one-hot vector of predicted class (shape = batch_size x 2) 
  * Convolution layers :

    | Layer Index | inputs | outputs| filter shape | stride | pooling-stride | activation | 
    | ----------- |:------:| ------:| ------------:| ------:| --------------:| ----------:|
    |      1      | 1      |  32    |     5x5      |   1    | max pooling - 2|   ReLU     |
    |      2      | 32     |  64    |     5x5      |   1    | max pooling - 2|   ReLU     |
    |      3      | 64     |  128   |     5x5      |   1    | max pooling - 2|   ReLU     |
  * Fully connected layer :
    * number of nodes : 1024
    * input : 10 x 10 x 128 (reshaped to ? X 1024)
    * output size : batch_size x 1024
    * activation : ReLU
  * An attempt to apply dropout is being done (more work on it comming soon)

### References
- Background on Satellite imaging https://www.kaggle.com/c/statoil-iceberg-classifier-challenge#Background
