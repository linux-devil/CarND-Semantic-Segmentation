## CarND-Semantic-Segmentation
SDCND - Semantic Segmentation based on FCNs 

## Overview

Semantic Segmentation is one of the high-level task that paves the way towards complete scene understanding. Objective of the project is to label the pixels of traffic data as road or non-road category. using the Fully Convolutional Network (FCN) described in the [Fully Convolutional Networks for Semantic Segmentation](https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf) by Jonathan Long, Even Shelhamer, and Trevor Darrel. The project is based on the starting project provided by Udacity in this [repo](https://github.com/udacity/CarND-Semantic-Segmentation).

![FCN](https://github.com/linux-devil/CarND-Semantic-Segmentation/blob/master/FCN1.png "FCN")


### Setup
#### GPU
main.py will check to make sure you are using GPU - if you don't have a GPU on your system, you can use AWS or another cloud computing platform.

#### Frameworks and Packages
Make sure you have the following is installed:

Python 3
TensorFlow
NumPy
SciPy

## Dataset
Project is using Kitti [Road Dataset](http://www.cvlibs.net/datasets/kitti/eval_road.php).

## Code Description

#### Main.py
Code is implemented in Main.py module which is responsible for various task such as layer extraction , implementing layers for FCN and training the neural network. 

#### load_vgg
In this method changes have been made to load specific layer tesnors which will be further used for implementing the layers for segmentation. Here we use "get_tensor_by_name" method and extract following layer tensors:
1. layer3_out
2. layer4_out
3. layer7_out

#### layers
In this method we create fully convolutional netowrk using upsampling and skip connections. Method returns the final output layer.

#### optimize
In this method we implemented loss and optimizer operations. Cross entropy loss is calculated using softmax_cross_entropy_with_logits after creating logits and loading correct labels.

#### train_nn
We train the neural network and print out the loss in this method. 

## Results
Following is the output of images after we trained our Segmentation model for 40 epochs on a GPU machine.


## Future Work
Over the period of time we have seen evolution in the Segmentation architecture. FCN is one of the first segmentation architecture but new architecture such as DeepLab and PSP networks are really promising and worth implementing. 

## References
[A guide to convolution arithmetic for deep
learning](https://arxiv.org/pdf/1603.07285.pdf)
[Guide to Semantic Segmentation with Deep Learning](http://blog.qure.ai/notes/semantic-segmentation-deep-learning-review#fcn)
