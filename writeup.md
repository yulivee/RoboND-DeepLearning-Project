# Project Follow-Me #

Welcome to the writeup of my attempt at the Follow-me Project! I am excited to share my findings with you!
My repository contains lots of weight collections, Please refer to the following files for my submission:

- `/data/weighths/config_model_weights_225`
- `/data/weighths/model_weights_225.h5`
- `model_training.html`
    

## Neural Network Architecture

In Project Follow-me, a Quadcopter with a camera should be able to follow a specific person (named „hero“ according to udacity documentation)  through a city with other inhabitants. So the task to solve is twofold:
1. Is the hero in the picture?
2. Where in the picture is the hero?

Question numer one qualifies a **Convolutional Neural Network** (Covnet or CNN) to be used. The advantage of covnets is, that the numbers of networkparameters can be drastically reduced by using knowledge about the structure of the data. As we are dealing with images for the task at hand, we know that it is not important to know in which corner of the image our hero is, to know the he is the hero. The weights are shared across all patches in an input layer.

Question number two makes it necessary to use a **Fully Convolutional Network** (FCN) , as this type of network preserves the spatial information in an image and enables us to get an understanding of the scene through semantic segmantation. In this type of Neural Network, the high-level reasoning layer is composed of *1x1 convolutional layers* instead of *fully connected layers*. This neat trick presevers the location information.

A typical FCN is composited of multiple **encoder** blocks, followed by 1x1 convolutional layers, followed by as many **decoder** blocks as there are encoder blocks. The encoder blocks extract features for the image. It is extracting more concrete features per layer: At the first, it might only look for basic shapes line lines and edges, next looking for circles and curves, and increasingly more complex shapes the deeper the layer.. The decoder blocks upscale the output back to the size of the original image. The result is a segmentation information for every pixel in the image. So called skip connections enable the network to use images from multiple resolution scales by connecting non-adjacent layers. This results in more precise semantic segmentation in the output image.


My final network layout looks like this:
![nwarch](https://github.com/yulivee/RoboND-DeepLearning-Project/raw/master/docs/network-drawing.png "Network Architecture")

I decided to go with 3 layers for encoder and decoder, as I wanted to use few layers. Networks with many layers are more prone to overfitting. After researching google for best practices in filter size and comming to the conclusion that it seemed common to gradually increase them, I decided to go with multiples of 8. My first attempt with some initial hyperparameters already went straigth for 31% final score, so I decided to stick with the architecture.

During the testing I noticed that I got stuck around ~38% testing, even varying the hyperparameters.
The best I could get was 0.398499510675, just below the magic mark...

So as a further attempt, I printed out the number of parameters with `model.summary` and got around 12,000 parameters with the provided testing data. I guess that this could be a too small number for properly recognizing features and decided to add an additional separable convolution layer to my decoder block function. Additionally I decided to make my model deeper to get more parameters and went for multiples of 36. This brough me to 250.000 parameters which sounded enough. Following the discussions on slack, I read up on Nesterov Momentum at cs231n.github.io/neural-networks-3/#baby and decided to give it a try. The result was a whopping 44 Grade, so ~6% better.

My final layout of my encoder and decoder blocks look like this:
![nwblocks](https://github.com/yulivee/RoboND-DeepLearning-Project/raw/master/docs/network-blocks.png "Network Block Structure")


## Neural Network Parameters

The Hyperparameters of this FCN are:

- Learning Rate
- Batch Size
- Epochs
- Steps per Epoch
- Validation Steps

I will explain my approach of tuning the parameters after the introduction of the inidividual parameters.

### Learning Rate
The learning rate is the amount of correction which the network applies when modifing the weights. Its an indicator how quickly the network can change its mind. This is usually a low value - I experimented with values between 0,01 and 0,002. Otherwise to much instability is introduced into the network.

I settled with a learning rate of 0,008

### Batch Size
As memory is a finite source, not all training input can be put into the network in one run. Therefore the input is divided into small subsets called batches. The input is randomly shuffled an then put into the batches.

I settled with a batch size of 32

### Epochs
An epoch is a full walkthrough of a neural network, that means performing a full forward- and backward-propagation. Performing this multiple times increases the network accuracy without the need for more testing data. This advantage will cease over time, there comes a point when the accuracy stops increasing. It is connected to the learning rate: smaller learning rates need more epochs to get a good accuracy. This makes sense as the changes are more subtle and therefore need more time to develop.

My best for epochs was 100, but it would probably work with a bit less as it didn't learn any more in the later epochs. I lacked the time and AWS Credit to do another test. If I would pursue this project further, I would test again with 70 epochs.

### Steps per Epoch
The steps per epoch is the number of training image batches which pass the network in 1 epoch. There is no need to put every image through the network in every epoch and not putting everything in everytime also helps with overfitting.

I settled with 130 steps per epoch ( which I calculated by training_images/ batch size )

### Validation Steps
The steps per epoch is the number of validation image batches which pass the network in 1 epoch. This is the same as steps per epoch with validation images.

I settled with 37 validation steps ( which I calculated by validation_images/ batch size )

### Hyperparameter Tuning Approach

I started with the default parameters in the jupyter notebook and started with 31% score. As I had trouble acquiring GPU Instances on Amazon and my computer proved to be quite slow with the learning process, I put the code from the notebook into a python script which I feed a json file with the various parameter configurations I wanted to try. This way, my computer could keep performing hyperparameter tuning while I could sleep or go to work. See training.py and training\_parameters.json. The script is called like this `sudo nice -n 19 ./training.py | tee training_output.log` (the nice is to up the priority of the script on linux).

The tuning was a kind of educated brute-force. I started with the default parameters and kept adjusting the learning rate. I then tried various epochs and started varying the steps per epoch. My best score to date is 0.444166837977% score using only the provided training images.

## Neural Network Layers

### 1x1 Convolutional Layers
A 1x1 convolutional layer is a mini-neural network of 1 pixel width and height. It is typically used between other convolution layers to increase the depth and number of parameters of a model without changing the structure of it. In the FCN used in the project, a convolutional layer is used between the encoder and decoder blocks instead of a fully connected layer to create maximum depth while preserving spatial information.


### Fully connected layers

In a fully connected layer all neurons have full connections to all activation functions from the previous layer. This is the common layer type in regular neural networks (as opposed to covnets). Their activations can be calculated with a matrix multiplication and an added bias. While this works great in calculating probabilities and answering yes-no questions, this does not work well with images of varying sizes: the size of the input is constrained by the size of the fully connected layer.

## Conclusion

In Theory, this model could work for following another object ( dog, cat, car etc.). The Convolutional layers are choosing and extracting features from the images on their own, so as long as there is a reasonably large set of training-data it could be trained to follow something else. It would probably need a bit of hyperparameter tuning. The limitiation I image is, that cars can be pretty indistiguishable - two yellow sedan taxis look pretty much the same apart from the license plate. So I imagine that it could be problematic to follow the correct car if there appear two cars of the same model. Adding additional layers could help identify more features but I guess it would be hard to fully eliminate that problem.

In a sense, the simulation was cheating a bit as well, because the hero was wearing a red shirt that no one else was wearing. But in real life, it is rare to find two people wearing exactly the same, having the same skintone, haircolor, height and body shape. There is more diversity in people than in cars, so the probability of collisions should be pretty small.

What was a bit of a disappoint, was that I was unable to test my result in the simulator, because the follower.py script kept core dumping. Myself, as well as the people in slack, did not find a solution so far. But as long as follower-script ran, the model worked nicely (although that was only for very short periods of time)

