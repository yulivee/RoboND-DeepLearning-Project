#! /home/yulivee/Projekte/Robotics/Nanodegree/miniconda3/envs/RoboND/bin/python
import os
import math
import glob
import json
import sys
import tensorflow as tf

from scipy import misc
import numpy as np

from tensorflow.contrib.keras.python import keras
from tensorflow.contrib.keras.python.keras import layers, models

from tensorflow import image

from utils import scoring_utils
from utils.separable_conv2d import SeparableConv2DKeras, BilinearUpSampling2D
from utils import data_iterator
#from utils import plotting_tools 
from utils import model_tools

def separable_conv2d_batchnorm(input_layer, filters, strides=1):
    output_layer = SeparableConv2DKeras(filters=filters,kernel_size=3, strides=strides,
                             padding='same', activation='relu')(input_layer)
    
    output_layer = layers.BatchNormalization()(output_layer) 
    return output_layer

def conv2d_batchnorm(input_layer, filters, kernel_size=3, strides=1):
    output_layer = layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, 
                      padding='same', activation='relu')(input_layer)
    
    output_layer = layers.BatchNormalization()(output_layer) 
    return output_layer

def bilinear_upsample(input_layer):
    output_layer = BilinearUpSampling2D((2,2))(input_layer)
    return output_layer    

def encoder_block(input_layer, filters, strides):
    
    # Create a separable convolution layer using the separable_conv2d_batchnorm() function.  
    output_layer = separable_conv2d_batchnorm(input_layer, filters, strides)
    
    return output_layer

def decoder_block(small_ip_layer, large_ip_layer, filters):
    
    # TODO Upsample the small input layer using the bilinear_upsample() function.
    output = bilinear_upsample(small_ip_layer)
    
    # TODO Concatenate the upsampled and large input layers using layers.concatenate
    concat_output = layers.concatenate([output, large_ip_layer])
    
    # TODO Add some number of separable convolution layers
    convolution_output_layer = separable_conv2d_batchnorm(concat_output, filters)
    output_layer = separable_conv2d_batchnorm(convolution_output_layer, filters)
    
    return output_layer

def fcn_model(inputs, num_classes):
    
    filters = 16
    strides = 2
    #Input size: 
    
    # Encoder Blocks. 
    # Remember that with each encoder layer, the depth of your model (the number of filters) increases.
    
    # Layer 1 size:
    layer_1 = encoder_block(inputs, filters, strides)
    # Layer 2 size:
    layer_2 = encoder_block(layer_1, filters*2, strides)
    # Layer 3 size:
    layer_3 = encoder_block(layer_2, filters*4, strides)
    
    
    # 1x1 Convolution layer using conv2d_batchnorm()
    # Convolution layer size:
    convolution_layer = conv2d_batchnorm(layer_3, filters*16, kernel_size=1, strides=1)
    
    # Decoder Blocks
    # Layer 4 size:
    layer_4 = decoder_block(convolution_layer, layer_2, filters*4)
    # Layer 5 size
    layer_5 = decoder_block(layer_4, layer_1, filters*2)
    # Layer 6 size:
    layer_6 = decoder_block(layer_5, inputs, filters)
    
    
    # The function returns the output layer of your model. "x" is the final layer obtained from the last decoder_block()
    return layers.Conv2D(num_classes, 1, activation='softmax', padding='same')(layer_6)

image_hw = 160
image_shape = (image_hw, image_hw, 3)
inputs = layers.Input(image_shape)
num_classes = 3

# Call fcn_model()
output_layer = fcn_model(inputs, num_classes)

with open('training_parameters.json') as json_data:
    input_args = json.load(json_data)

for run, run_number in input_args.items():    
    
    print("----- Run: "+run+" --------------------------------------------------------------------")
    learning_rate = float(run_number['learning_rate'])
    print("Learning Rate: "+str(learning_rate))
    batch_size = int(run_number['batch_size'])
    print("Batch Size: "+str(batch_size))
    num_epochs = int(run_number['num_epochs'])
    print("Epochs: "+str(num_epochs))
    steps_per_epoch = int(run_number['steps_per_epoch'])
    print("Steps per Epoch: "+str(steps_per_epoch))
    validation_steps = int(run_number['validation_steps'])
    print("Validation Steps: "+str(validation_steps))
    workers = int(run_number['workers'])
    print("Workers: "+str(workers))
    print("--------------------------------------------------------------------------------")
    
    # Define the Keras model and compile it for training
    model = models.Model(inputs=inputs, outputs=output_layer)
    
    model.compile(optimizer=keras.optimizers.Nadam(learning_rate), loss='categorical_crossentropy')
    
    # Data iterators for loading the training and validation data
    train_iter = data_iterator.BatchIteratorSimple(batch_size=batch_size,
                                                   data_folder=os.path.join('..', 'data', 'train'),
                                                   image_shape=image_shape,
                                                   shift_aug=True)
    
    val_iter = data_iterator.BatchIteratorSimple(batch_size=batch_size,
                                                 data_folder=os.path.join('..', 'data', 'validation'),
                                                 image_shape=image_shape)
    
    #logger_cb = plotting_tools.LoggerPlotter()
    #callbacks = [logger_cb]
    callbacks = []
    
    model.fit_generator(train_iter,
                        steps_per_epoch = steps_per_epoch, # the number of batches per epoch,
                        epochs = num_epochs, # the number of epochs to train for,
                        validation_data = val_iter, # validation iterator
                        validation_steps = validation_steps, # the number of batches to validate on
                        callbacks=callbacks,
                        workers = workers) 
    
    weight_file_name = 'model_weights_'+run
    model_tools.save_network(model, weight_file_name)
    
    run_num = 'run_'+run
    
    val_with_targ, pred_with_targ = model_tools.write_predictions_grade_set(model,
                                            run_num,'patrol_with_targ', 'evaluation') 
    
    val_no_targ, pred_no_targ = model_tools.write_predictions_grade_set(model, 
                                            run_num,'patrol_non_targ', 'evaluation') 
    
    val_following, pred_following = model_tools.write_predictions_grade_set(model,
                                            run_num,'following_images', 'evaluation')
    
    print("--------------------------------------------------------------------------------")
    print( "Scores for while the quad is following behind the target" )                                    
    true_pos1, false_pos1, false_neg1, iou1 = scoring_utils.score_run_iou(val_following, pred_following)
    print( "true_pos1: "  + str(true_pos1)  )
    print( "false_pos1: " + str(false_pos1) )
    print( "false_neg1: " + str(false_neg1) )
    print( "iou1: "       + str(iou1)       )
    
    print("--------------------------------------------------------------------------------")
    print( "Scores for while the quad is on patrol and the target is not visable" )                                    
    true_pos2, false_pos2, false_neg2, iou2 = scoring_utils.score_run_iou(val_no_targ, pred_no_targ)
    print( "true_pos2: "  + str(true_pos2)  )
    print( "false_pos2: " + str(false_pos2) )
    print( "false_neg2: " + str(false_neg2) )
    print( "iou2: "       + str(iou2)       )
    
    print("--------------------------------------------------------------------------------")
    print( "This score measures how well the neural network can detect the target from far away" )
    true_pos3, false_pos3, false_neg3, iou3 = scoring_utils.score_run_iou(val_with_targ, pred_with_targ)
    print( "true_pos3: "  + str(true_pos3)  )
    print( "false_pos3: " + str(false_pos3) )
    print( "false_neg3: " + str(false_neg3) )
    print( "iou3: "       + str(iou3)       )
    
    print("--------------------------------------------------------------------------------")
    print( "Summation of Scores" )
    true_pos = true_pos1 + true_pos2 + true_pos3
    false_pos = false_pos1 + false_pos2 + false_pos3
    false_neg = false_neg1 + false_neg2 + false_neg3
    print( "true_pos: "  + str(true_pos)  )
    print( "false_pos: " + str(false_pos) )
    print( "false_neg: " + str(false_neg) )
    
    print("--------------------------------------------------------------------------------")
    print( "Sum all the true positives, etc from the three datasets to get a weight for the score" )
    weight = true_pos/(true_pos+false_neg+false_pos)
    print( "Weight: "+ str(weight) )
    
    print("--------------------------------------------------------------------------------")
    print( "The IoU for the dataset that never includes the hero is excluded from grading" )
    final_IoU = (iou1 + iou3)/2
    print( "Final IoU: "+ str(final_IoU) )
    
    print("--------------------------------------------------------------------------------")
    print( "Final Grade Score" )
    final_score = final_IoU * weight
    print( "Grade: "+ str(final_score) )
