#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 21 12:51:15 2018

@author: vlad
"""


import os
os.environ['MKL_THREADING_LAYER'] = 'GNU'
os.environ['KERAS_BACKEND'] = 'tensorflow'
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

import csv
import pandas as pd
from PIL import Image
import random

from keras.preprocessing.image import ImageDataGenerator
import keras.preprocessing.image as kim



#SELECT ALL IDS WHERE NUMBER OF CORRESPONDING IMAGES IS EQUAL TO OR EXCEEDS A SET 
#MIN AND MAX THRESHOLDS
#EXCLUDE new_whale
def ds_search_items(img_count_min = 8, img_count_max = 8):

    filtered_count_dict = count_dict[(count_dict['Image'] >= img_count_min) & (count_dict['Image'] <= 
                           img_count_max) & (count_dict['Image'] != 
                           new_whale)]
    filtered_count_dict = filtered_count_dict.to_dict(orient = 'index')
    search_items = list(filtered_count_dict.keys())
    
    return search_items


#MODEL TO COMPUTE PREDICTED EMBEDDINGS; IT DOESN'T GET TRAINED
def WhaleModel(input_shape=(224, 224, 3)):

    in_X = Input(shape =(224, 224, 3))

    shared_dense1 = Dense(4096, activation='relu', name='fc1')
    shared_dense2 = Dense(4096, activation='relu', name='fc2')
    shared_dense3 = Dense(1024, name='predictions')
    shared_L2norm = Lambda(lambda  x: K.l2_normalize(x,axis=1))

    #baseModel is defined in New_whaleModel as VGG16 with disabled top layers
    X = baseModel(in_X)
    X = shared_dense1(X)
    X = shared_dense2(X)
    X = shared_dense3(X)
    X = shared_L2norm(X)

    #create Keras model instance
    model = Model(inputs = in_X, outputs = X, name='WhaleModel')

    return model



#FUNCTION TO GET IMAGE PATH DEPENDING ON TYPE OF IMAGE
def get_img_path(test_item, new_whale = False, img_to_store = False):

    '''
    TAKE ONE INDIVIDUAL test_item, new_whale - whether a whale is idenfied as know in train
    database; img_to_store - whether this particular image will be made part of the lookup database
    RETURN image path 
    '''  
    #search_whale.set_index(keys = 'Id', inplace = True)
    if new_whale == False and img_to_store == True:
        img_list = search_whale.loc[search_whale['Id'] == test_item]
        img_path = '/home/vlad/Documents/Whales/train/' + img_list.iloc[0,0]
    elif new_whale == False and img_to_store == False:
        img_list = search_whale.loc[search_whale['Id'] == test_item]
        img_path = '/home/vlad/Documents/Whales/train/' + img_list.iloc[1,0]
    else:    
        img_list = search_whale.loc[search_whale['Id'] == 'new_whale']
        img_path = '/home/vlad/Documents/Whales/train/' + img_list.iloc[random.choice(range(len(img_list))),0]

    return img_path


#ENCODE IMAGE INTO A PREDICTED VALUE USING WHALEMODEL.PREDICT TO BE PASSED AS AN ARGUMENT
def img_to_encoding(image_path, model):
    
    img_e = image.load_img(image_path, target_size=(224, 224))
    x_train = image.img_to_array(img_e)
    x_train = np.expand_dims(x_train, axis=0)
    x_train = preprocess_input(x_train)
    
    embedding = model.predict(x_train)
    return embedding


#IDENTIFY WHALE BY ID AND STATE WHETHER IT'S IN THE DATABASE
def id_whale_silent(image_path, database, model):
    """
    Implements whale recognition by finding whether the image is of a known whale
    
    
    Arguments:
    image_path -- path to an image
    database -- database containing image encodings along with the name of the person on the image
    model -- model instance in Keras
    
    Returns:
    min_dist -- the minimum distance between image_path encoding and the encodings from the database
    identity -- string, the name prediction for the person on image_path
    """
    
    encoding = img_to_encoding(image_path, model)
       
    # Initialize "min_dist" to a large value, say 100
    min_dist = 100
    
    # Loop over the database dictionary's names and encodings
    for (name, db_enc) in database.items():
        
        # Compute L2 distance between the target "encoding" and the current "emb" from the database
        dist = np.linalg.norm(encoding - db_enc)

        # If this distance is less than the min_dist, then set min_dist to dist, and identity to name 
        if dist < min_dist:
            min_dist = dist
            identity = name
    
    return min_dist, identity


#test percentage of positive images identified correctly
def test_positive_build_output():
    '''
    Based on predictions for test items, matches actual whale ids with predictions
    and returns the % of correct matches
    '''
    identities = []
    distances= []

    for item in range(len(test_items)):
    
        test_img_path = get_img_path(test_items[item], new_whale = False, img_to_store = False)
        distance, identity = id_whale_silent(test_img_path, database, whaleModel)
        identities.append(identity)
        distances.append(distance)
    
    results = list(zip(test_items, identities, distances))
    results = pd.DataFrame(results)
    results.columns = ['actual', 'predict', 'distance']
    results['match'] = results['actual'] == results['predict']
    result_score = sum(results['match'])/len(results)
    
    print('correctly matched %f' % (result_score))
    return result_score
        

#Triplet loss function
def triplet_loss(y_true, values_pred, alpha = 0.5):
    """
    Implementation of the triplet loss
    
    Arguments:
    y_true -- true labels, required when you define a loss in Keras, you don't need it in this function.
    y_pred -- python list containing three objects:
            anchor -- the encodings for the anchor images, of shape (None, 128)
            positive -- the encodings for the positive images, of shape (None, 128)
            negative -- the encodings for the negative images, of shape (None, 128)
    
    Returns:
    loss -- real number, value of the loss
    """

    # process batches consisting of predicted anchor, positive and negative results
    l = K.int_shape(values_pred)[1]/3
    l = tf.cast(l, tf.int32)
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


#This is a pre-trained VGG16 model for low-level features
def BaseModel(input_shape):
    """
    Implementation of the model.
    
    Arguments:
    input_shape -- shape of the images of the dataset

    Returns:
    model -- a Model() instance in Keras
    """   
    # Define the input placeholder as a tensor with shape input_shape.This is an input image
    X_input = Input(input_shape)


    #Get back the convolutional part of a VGG network trained on ImageNet
    base_model = VGG16(weights='imagenet', include_top=False, input_shape = (224, 224, 3))
    #base_model.summary()

    # first: train only the top layers (which were randomly initialized)
    # i.e. freeze all convolutional InceptionV3 layers
    for layer in base_model.layers[:7]:
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
shared_dense3 = Dense(1024, name='predictions')
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

#Compile the model to configure the learning process
whaleModel_fuse.compile(optimizer = "Adam", loss = triplet_loss)


#This function selects random positive and negative images for a given anchor
def select_p_n_images(item):
    
     whale_name = search_whale.loc[search_whale['Image'] == item].iloc[0,1]             
     
     Pos = search_whale.loc[(search_whale['Id'] == whale_name) & (search_whale['Image'] != item)]
     Neg = search_whale.loc[search_whale['Id'].isin(search_items)]
     Neg = Neg.loc[search_whale['Id'] != whale_name]
            
     Pos = list(Pos['Image'])
     Neg = list(Neg['Image'])

     img_p = random.choice(Pos)
     img_n = random.choice(Neg)
    
     return img_p, img_n



#create lists of positive and negative image names
def create_p_n_lists():
    
    list_p = []
    list_n = []    

    for item in search_whale_img_list:

        img_p, img_n = select_p_n_images(item)
        
        list_p.append(img_p)
        list_n.append(img_n)
        
    return list_p, list_n


#get image from file and convert into array
def build_image_X_dataset(search_list):
    
    X = np.empty((len(search_list),224,224,3))
    rotationSize = 25
    zoomRangeWidth = .8
    zoomRangeHeight = 1.1    
    widthRange = .2
    heightRange = .2
    
    for i, ID in enumerate(search_list):
        
        img = image.load_img('/home/vlad/Documents/Whales/train/' + ID, target_size=(224, 224))
        #img = np.array(img)
        #img = kim.random_rotation(img, rotationSize, row_axis=0, col_axis=1,
        #                          channel_axis=2, fill_mode='nearest')
        #img = kim.random_zoom(img, zoom_range=(zoomRangeWidth,zoomRangeHeight),
        #                      row_axis=0,col_axis=1, fill_mode='nearest')
        #img = kim.random_shift(img,wrg=widthRange, hrg=heightRange,row_axis=0,
        #                       col_axis=2, channel_axis=0, fill_mode='nearest')
        img = image.img_to_array(img)
        X[i,] = img 
    
    return X


#Model to compute predicted embeddings; it does not get trained
def WhaleModel(input_shape=(224, 224, 3)):

    in_X = Input(shape =(224, 224, 3))

    shared_dense1 = Dense(4096, activation='relu', name='fc1')
    shared_dense2 = Dense(4096, activation='relu', name='fc2')
    shared_dense3 = Dense(1024, name='predictions')
    shared_L2norm = Lambda(lambda  x: K.l2_normalize(x,axis=1))

    #baseModel is defined in New_whaleModel as VGG16 with disabled top layers
    X = baseModel(in_X)
    X = shared_dense1(X)
    X = shared_dense2(X)
    X = shared_dense3(X)
    X = shared_L2norm(X)

    #create Keras model instance
    model = Model(inputs = in_X, outputs = X, name='WhaleModel')

    return model

#just need to compile the model with weights from whaleModel_fuse that can accept input
#save model into h5 and load weights into whaleModel
#whaleModel_fuse.save('/Users/vlad/Projects/whales/whales_model/whaleModel_fuse.h5')

whaleModel = WhaleModel(input_shape=(224, 224, 3))
#whaleModel.load_weights('/home/vlad/Documents/Whales/whales_model/whaleModel_fuse_test.h5', by_name=True)
#baseModel.load_weights('/home/vlad/Documents/Whales/whales_model/baseModel_test.h5', by_name=True)
#embedding_test is just to make sure the embeddings get changed with training, i.e. the model


#build image generator augmenting images  in real time
gen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=15,
    zoom_range=[0.8,1.1])

gen = ImageDataGenerator()

#Data generator that simultaneously builds anchor, positive and negative data streams
def gen_flow_three_inputs(X1, X2, X3, y):
    
    genX1 = gen.flow(X1,y, batch_size=batch_size, seed = 1)
    genX2 = gen.flow(X2,y, batch_size=batch_size, seed = 1)
    genX3 = gen.flow(X3,y, batch_size=batch_size, seed = 1)
    
    while True:
        X1i = genX1.next()
        X2i = genX2.next()
        X3i = genX3.next()
        yield [X1i[0], X2i[0], X3i[0]], X1i[1]

#______________________________________________________________________________
#BUILD WHALE/IMAGE DICTIONARIES Specify the file to be opened, and use the ‘rb’ method meaning “read binary”
search_whale = pd.read_csv('/home/vlad/Documents/Whales/train.csv')
count_dict = pd.pivot_table(search_whale, index = 'Id', aggfunc = 'count')


#get number of images for new whale
new_whale = count_dict.loc['new_whale','Image']


#Search for necessary items depending on the count of individual whales. this becomes train data set
search_items = ds_search_items(img_count_min = 2, img_count_max = 34)


#Select training list of whales with above thresholds and exclude new_whale
search_whale_img_list = search_whale.loc[search_whale['Id'].isin(search_items)] 
search_whale_img_list = search_whale_img_list['Image']
search_whale_img_list = search_whale_img_list.values.tolist()


#Search for items for the validation data set
test_items = ds_search_items(img_count_min = 2, img_count_max = 2)




#Upload weights if necessary
whaleModel_fuse.load_weights('/home/vlad/Documents/Whales/whales_model/whaleModel_fuse_test.h5', by_name=True)
baseModel.load_weights('/home/vlad/Documents/Whales/whales_model/baseModel_test.h5', by_name=True)

#build anchor. It doesn't get changed, so only do it once. Build pos and nev in loop below
A_img = build_image_X_dataset(search_whale_img_list)

#This picks up random matching positive and negative images
list_p, list_n = create_p_n_lists()

#build anchor, positive and negative datasets and y-train (just filled with ones, it's not used)
P_img = build_image_X_dataset(list_p)
N_img = build_image_X_dataset(list_n)
y_train = np.ones(len(A_img)).reshape(len(A_img),1)



# create generator
gen_flow = gen_flow_three_inputs(A_img, P_img, N_img, y_train)

#set model training parameters
num_repeats = 5
batch_size = 64
epochs = 5

test_outputs = []


for i in range(num_repeats):
    
    whaleModel_fuse.fit_generator(gen_flow, steps_per_epoch=len(A_img)/batch_size, epochs = epochs)
    
    #save weight
    whaleModel_fuse.save('/home/vlad/Documents/Whales/whales_model/whaleModel_fuse_test.h5')
    baseModel.save('/home/vlad/Documents/Whales/whales_model/baseModel_test.h5')
    
    #load weights for testing
    whaleModel.load_weights('/home/vlad/Documents/Whales/whales_model/whaleModel_fuse_test.h5', by_name=True)
    
    #Build database of validation whales using known images
    database = {}
    for i in range(len(test_items)):
        img_path = get_img_path(test_items[i], new_whale = False, img_to_store = True)
        database[test_items[i]] = img_to_encoding(img_path, whaleModel)
    
    test_outputs.append(test_positive_build_output())



    
    