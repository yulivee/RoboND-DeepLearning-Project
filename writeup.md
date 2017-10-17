# Project Follow-Me #

Welcome to the writeup of my attempt at the Follow-me Project! I am excited to share my findings with you!

## Neural Network Architecture

In Project Follow-me, a Quadcopter with a camera should be able to follow a specific person (named „hero“ according to udacity documentation)  through a city with other inhabitants. So the task to solve is twofold:
1. Is the hero in the picture?
2. Where in the picture is the hero?

Question numer one qualifies a **Convolutional Neural Network** (Covnet or CNN) to be used. The advantage of covnets is, that the numbers of networkparameters can be drastically reduced by using knowledge about the structure of the data. As we are dealing with images for the task at hand, we know that it is not important to know in which corner of the image our hero is, to know the he is the hero. The weights are shared across all patches in an input layer.

Question number two makes it necessary to use a **Fully Convolutional Network** (FCN) , as this type of network preserves the spatial information in an image and enables us to get an understanding of the scene through semantic segmantation. In this type of Neural Network, the high-level reasoning layer is composed of *1x1 convolutional layers* instead of a *fully connected layers*. This neat trick presevers the location information.

A typical FCN is composited of multiple **encoder** blocks, followed by 1x1 convolutional layers, followed by as many **decoder** blocks as there are encoder blocks. The encoder blocks extract features for the image, the decoder blocks upscale the output back to the size of the original image. The result is a segmentation information for every pixel in the image. So called skip connections enable the network to use images from multiple resolution scales by connecting non-adjacent layers. This results in more precise semantic segmentation in the output image.

My final network layout looks like this:
![nwarch]ttps://github.com/yulivee/RoboND-DeepLearning-Project/raw/master/docs/network-drawing.png "Network Architecture")

I decided to go with 3 layers for encoder and decoder, as I wanted to use few layers. Networks with many layers are more prone to overfitting. After researching google for best practices in filter size and comming to the conclusion that it seemed common to gradually increase them, I decided to go with multiples of 8.

## Neural Network Parameters

## Neural Network Layers

## Image Manipulation
