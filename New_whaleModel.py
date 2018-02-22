#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 18 13:30:32 2018

@author: vlad
"""

from keras import backend as K
import tensorflow as tf
from keras.layers.core import Lambda, Flatten, Dense
from keras.layers import Concatenate, concatenate
import numpy as np
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.models import Model
from keras.applications.vgg16 import VGG16
from keras.utils import np_utils
from keras.layers import Reshape
from keras.preprocessing import image
from PIL import Image
from keras.applications.resnet50 import preprocess_input, decode_predictions



def BaseModel(input_shape):
    """
    Implementation of the model.
    
    Arguments:
    input_shape -- shape of the images of the dataset

    Returns:
    model -- a Model() instance in Keras
    """   
    # Define the input placeholder as a tensor with shape input_shape. Think of this as your input image!
    X_input = Input(input_shape)


    #Get back the convolutional part of a VGG network trained on ImageNet
    base_model = VGG16(weights='imagenet', include_top=False, input_shape = (224, 224, 3))
    #base_model.summary()

    # first: train only the top layers (which were randomly initialized)
    # i.e. freeze all convolutional InceptionV3 layers
    for layer in base_model.layers:
        layer.trainable = False

    #Use the generated model 
    output_base_model = base_model(X_input)

    #Add the fully-connected top trainable layers 
    X = Flatten(name='flatten')(output_base_model)
   
    model = Model(inputs = X_input, outputs = X, name='BaseModel')
    
    return model

    
baseModel = BaseModel(input_shape=(224, 224, 3)) 



    #Add the fully-connected top trainable layers 

shared_dense1 = Dense(4096, activation='relu', name='fc1')
shared_dense2 = Dense(4096, activation='relu', name='fc2')
shared_dense3 = Dense(121, name='predictions')
shared_L2norm = Lambda(lambda  x: K.l2_normalize(x,axis=1))


in_a = Input(shape =(224, 224, 3))
in_p = Input(shape =(224, 224, 3))
in_n = Input(shape =(224, 224, 3))


out_a = baseModel(in_a)
out_p = baseModel(in_p)
out_n = baseModel(in_n)


out_a = shared_dense1(out_a)
out_a = shared_dense2(out_a)
out_a = shared_dense3(out_a)
out_a = shared_L2norm(out_a)

out_p = shared_dense1(out_p)
out_p = shared_dense2(out_p)
out_p = shared_dense3(out_p)
out_p = shared_L2norm(out_p)


out_n = shared_dense1(out_n)
out_n = shared_dense2(out_n)
out_n = shared_dense3(out_n)
out_n = shared_L2norm(out_n)


values_pred = concatenate([out_a, out_p, out_n], axis = 1)


whaleModel_fuse = Model([in_a, in_p, in_n], values_pred)

#Step 2: compile the model to configure the learning process
whaleModel_fuse.compile(optimizer = "Adam", loss = triplet_loss)


#Step 3: train the model
whaleModel_fuse.fit(x = [Anchor, Positive, Negative], y = y_train, epochs = 1)



def triplet_loss(y_true, values_pred, alpha = 0.2):
    """
    Implementation of the triplet loss as defined by formula (3)
    
    Arguments:
    y_true -- true labels, required when you define a loss in Keras, you don't need it in this function.
    y_pred -- python list containing three objects:
        !!!! CHECK SHAPE - THIS IS OLD SHAPE
            anchor -- the encodings for the anchor images, of shape (None, 128)
            positive -- the encodings for the positive images, of shape (None, 128)
            negative -- the encodings for the negative images, of shape (None, 128)
    
    Returns:
    loss -- real number, value of the loss
    """
    # process batches consisting of predicted anchor, positive and negative results
    l = K.int_shape(values_pred)[1]/3
    anchor, positive, negative = values_pred[: ,0:l], values_pred[: ,l:l*2], values_pred[: ,l*2:l*3]
   
    # Step 1: Compute the (encoding) distance between the anchor and the positive
    pos_dist = tf.reduce_sum(tf.square(anchor - positive), axis = -1)
    
    # Step 2: Compute the (encoding) distance between the anchor and the negative
    neg_dist = tf.reduce_sum(tf.square(anchor - negative), axis = -1)

    # Step 3: subtract the two previous distances and add alpha.
    basic_loss = tf.add(tf.subtract(pos_dist, neg_dist), alpha)

    # Step 4: Take the maximum of basic_loss and 0.0. Sum over the training examples.
    loss = tf.reduce_sum(tf.maximum(basic_loss ,0))

    
    return loss




#THIS FUNCTION  TAKES AVAILABLE IMAGE FILES AND TURNS THEM INTO ANCHOR, POSITIVE AND
#NEGATIVE FOR TRAINING
def generate_arrays_from_files_BATCH(start = 0, batch_size = 327):
         
        n_batches = len(search_whale_img_list)/batch_size
    
     
        count = 0  
        for item in search_whale_img_list[start:(start+batch_size)]:
            
            
            # create Numpy arrays of input data
            # anchor, positive and negative
            img_a = image.load_img('/Users/vlad/Projects/whales/train/' + item, target_size=(224, 224))

            Anchor_array = image.img_to_array(img_a)
            Anchor_array = np.expand_dims(Anchor_array, axis=0)
            Anchor_array = preprocess_input(Anchor_array)
            
                            
            whale_name = search_whale.loc[search_whale['Image'] == item, 'Id'][0]             
            Pos = search_whale.loc[(search_whale['Id'] == whale_name) & (search_whale['Image'] != item)]
            Neg = search_whale.loc[search_whale['Id'] != whale_name]
            
            Pos = list(Pos.index.values)
            Neg = list(Neg.index.values)
            
            img_p = image.load_img('/Users/vlad/Projects/whales/train/' + random.choice(Pos), target_size=(224, 224))
            Positive_array = image.img_to_array(img_p)
            Positive_array = np.expand_dims(Positive_array, axis=0)
            Positive_array = preprocess_input(Positive_array)
            
                        
            img_n = image.load_img('/Users/vlad/Projects/whales/train/' + random.choice(Neg), target_size=(224, 224))
            Negative_array = image.img_to_array(img_n)
            Negative_array = np.expand_dims(Negative_array, axis=0)
            Negative_array = preprocess_input(Negative_array)
            
            if count == 0:
                
                Anchor = Anchor_array
                Positive = Positive_array
                Negative = Negative_array
            else:
            
                Anchor = np.concatenate((Anchor, Anchor_array), axis = 0)
                Positive = np.concatenate((Positive, Positive_array), axis = 0)
                Negative = np.concatenate((Negative, Negative_array), axis = 0)
                
            count += 1
        
        return [Anchor, Positive, Negative]   
        
 
    
#TRAIN MODEL ON BATCHES PASSED BY generate_arrays_from_files_BATCH FUNCTION
def train_model():
    
    start = 0 
    batch_size = 327
    n_batches = len(search_whale_img_list)/batch_size
    
    for n in range(n_batches):
    
        Anchor, Positive, Negative = generate_arrays_from_files_BATCH(start = start, batch_size = batch_size)
        y_train = np.ones(len(Anchor)).reshape(len(Anchor),1)
        whaleModel_fuse.fit(x = [Anchor, Positive, Negative], y = y_train, epochs = 40, batch_size = 32)
        
        start = start + batch_size
    
        whaleModel_fuse.save('/Users/vlad/Projects/whales/whales_model/whaleModel_fuse_test.h5')
        
        print(n)


train_model()

