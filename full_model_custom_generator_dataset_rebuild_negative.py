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

import random

from keras.preprocessing.image import ImageDataGenerator
import keras.preprocessing.image as kim
from keras import optimizers
import itertools
from shutil import copyfile
from random import shuffle




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
    identity = ''
    
    # Loop over the database dictionary's names and encodings
    for (name, db_enc) in database.items():
        
        # Compute L2 distance between the target "encoding" and the current "emb" from the database
        dist = np.linalg.norm(encoding - db_enc)

        # If this distance is less than the min_dist, then set min_dist to dist, and identity to name 
        if dist < min_dist:
            min_dist = dist
            identity = name
    
    return min_dist, identity


def select_closest_negatives(image_path, test_item, database, model):
    
    '''
    similar to id_whale_silent, but instead of picking one item with minimum
    distance, the fuction selects 5 items closest to the embedding and returns
    them to be used as negatives in the following training iteration.
    Arguments:
    test_item
    image_path = path to test item
    database
    '''

    encoding = img_to_encoding(image_path, model)
    dist_list = []
    name_list = []

    for (name, db_enc) in database.items():
        
        # Compute L2 distance between the target "encoding" and the current "emb" from the database
        dist = np.linalg.norm(encoding - db_enc)
        dist_list.append(dist)
        name_list.append(name)
        
    closest_list = list(zip(name_list, dist_list))
    closest_negatives = [item for item in closest_list if item[0] != test_item]
    closest_negatives.sort(key = lambda x: x[1])
    closest_negatives = closest_negatives[:10]
    closest_negatives, _ = zip(*closest_negatives)
    
    return list(closest_negatives)



def get_match_index(image_path, test_item, database, model):
    
    '''
    similar to id_whale_silent, but instead of picking one item with minimum
    distance, the fuction selects 5 items closest to the embedding and returns
    them to be used as negatives in the following training iteration.
    Arguments:
    test_item
    image_path = path to test item
    database
    '''

    encoding = img_to_encoding(image_path, model)
    dist_list = []
    name_list = []

    for (name, db_enc) in database.items():
        
        # Compute L2 distance between the target "encoding" and the current "emb" from the database
        dist = np.linalg.norm(encoding - db_enc)
        dist_list.append(dist)
        name_list.append(name)
        
    closest_list = list(zip(name_list, dist_list))
    closest_list.sort(key = lambda x: x[1])
    closest_list, _ = zip(*closest_list)
    closest_index = closest_list.index(test_item)
    
    return closest_index



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
    for layer in base_model.layers[:25]:
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
shared_dropout = Dropout(0.6, name='drop')
shared_dense3 = Dense(1024, name='predictions')
#shared_L2norm = Lambda(lambda  x: K.l2_normalize(x,axis=1))


in_a = Input(shape =(224, 224, 3))
in_p = Input(shape =(224, 224, 3))
in_n = Input(shape =(224, 224, 3))


out_a = baseModel(in_a)
out_p = baseModel(in_p)
out_n = baseModel(in_n)


out_a = shared_dense1(out_a)
out_a = shared_dense2(out_a)
out_a = shared_dropout(out_a)
out_a = shared_dense3(out_a)
#out_a = shared_L2norm(out_a)

out_p = shared_dense1(out_p)
out_p = shared_dense2(out_p)
out_p = shared_dropout(out_p)
out_p = shared_dense3(out_p)
#out_p = shared_L2norm(out_p)


out_n = shared_dense1(out_n)
out_n = shared_dense2(out_n)
out_n = shared_dropout(out_n)
out_n = shared_dense3(out_n)
#out_n = shared_L2norm(out_n)


values_pred = concatenate([out_a, out_p, out_n], axis = 1)


whaleModel_fuse = Model([in_a, in_p, in_n], values_pred)

#Compile the model to configure the learning process
whaleModel_fuse.compile(optimizer = "Adam", loss = triplet_loss)
#for fine-tuning convolutional layers only
#whaleModel_fuse.load_weights('/home/vlad/Documents/Whales/whales_model/whaleModel_fuse_test.h5', by_name=True)
#whaleModel_fuse.compile(optimizer = optimizers.SGD(lr=1e-4, momentum = 0.9), loss = triplet_loss)

#Model to compute predicted embeddings; it does not get trained
def WhaleModel(input_shape=(224, 224, 3)):

    in_X = Input(shape =(224, 224, 3))

    shared_dense1 = Dense(4096, activation='relu', name='fc1')
    shared_dense2 = Dense(4096, activation='relu', name='fc2')
    shared_dropout = Dropout(0.6, name='drop')
    shared_dense3 = Dense(1024, name='predictions')
    #shared_L2norm = Lambda(lambda  x: K.l2_normalize(x,axis=1))

    #baseModel is defined in New_whaleModel as VGG16 with disabled top layers
    X = baseModel(in_X)
    X = shared_dense1(X)
    X = shared_dense2(X)
    X = shared_dropout(X)
    X = shared_dense3(X)
    #X = shared_L2norm(X)

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




def get_whale_images(whale_id):
    '''
    Agruments:
        item = individual whale id
    Returns:
        list of corresponding images'''
        
    whale_images = search_whale[search_whale['Id'] == whale_id]
    
    return list(whale_images['Image'])
    
    

#returns a list of anchor and positive tuples
def create_a_p_permutations(item):
    
    a_p_list = list(itertools.permutations(item, 2))
    return a_p_list



#returns randomly selected negative whale image items
def create_n_list(item, anchor):
    '''
    Arguments:
        item = individual whale id
        anchor = anchor list for individual whale, only used for length
    '''
    
    n_list = search_whale[search_whale['Id'] != item]
    n_list = list(n_list['Image'])
    rand_n = list(range(0,len(n_list)))
    rand_n = np.random.choice(rand_n, size=len(anchor), replace=False)
    n = [n_list[i] for i in rand_n]
    
    return n


def create_n_list_cycle(item, anchor):
    '''
    Arguments:
        item = individual whale id
        anchor = anchor list for individual whale, only used for length
    '''
    
    n_list = closest_negatives[item][1]
    rand_n = list(range(0,len(n_list)))
    rand_n = np.random.choice(rand_n, size=len(anchor), replace=True)
    n = [n_list[i] for i in rand_n]
    
    return n


#returns list of image triplets randomly selected
def select_img_with_threshold(APN_list, threshold = 30):
    
    if len(APN_list) > threshold:
        range_list = list(range(0,len(APN_list)))    
        select_ind = np.random.choice(range_list, size = threshold, replace = False)
        APN_list = [APN_list[i] for i in select_ind]
    
    return APN_list
    

def image_augmentation(img):
    '''
    Arguments:
        img = list of np.arrays of images
    Returns:
        list of arrays after transformations
    '''
    rotationSize = 15
    zoomRangeWidth = .8
    zoomRangeHeight = 1.1
    intensity = 15    
    widthRange = .1
    heightRange = .1

    img = [kim.random_rotation(item, rotationSize, row_axis=0, col_axis=1,
                channel_axis=2, fill_mode='nearest') for item in img]
    img = [kim.random_shift(item, wrg=widthRange, hrg=heightRange,row_axis=0,
                col_axis=1, channel_axis=2, fill_mode='nearest') for item in img]
    img = [kim.random_zoom(item, zoom_range=(zoomRangeWidth,zoomRangeHeight),
                row_axis=0,col_axis=1, channel_axis=2, fill_mode='nearest') for item in img]
    img = [kim.random_shear(item, intensity=intensity,
                row_axis=0,col_axis=1, channel_axis=2, fill_mode='nearest') for item in img]
    
    return img
    

def generator_from_df(df, batch_size):
    '''
    generator that yields A-P-N and Y
    '''
    
    nbatches, n_skipped_per_epoch = divmod(df.shape[0], batch_size)
    
    count = 1
    epoch = 0
    
    while 1:
        
        #shuffle dataframe
        df = df.sample(frac = 1)
        
        epoch += 1
        
        i,j = 0, batch_size
        
        #Mini_batches completed within epoch
        mini_batches_completed = 0
        
        for _ in range(nbatches):
            
            sub = df.iloc[i:j]
            
            #preprocess input
            img = [image.load_img(f, target_size = (224,224)) for f in sub['anchor']]
            img = [np.array(item) for item in img]   
            img = image_augmentation(img)
            A = np.array([(2 * image.img_to_array(item)/255.0 - 0.5) for item in img])


            img = [image.load_img(f, target_size = (224,224)) for f in sub['positive']]
            img = [np.array(item) for item in img]   
            img = image_augmentation(img)
            P = np.array([(2 * image.img_to_array(item)/255.0 - 0.5) for item in img])


            img = [image.load_img(f, target_size = (224,224)) for f in sub['negative']]
            img = [np.array(item) for item in img]   
            img = image_augmentation(img)
            N = np.array([(2 * image.img_to_array(item)/255.0 - 0.5) for item in img])

            Y = sub['label'].values   
                
            mini_batches_completed += 1
                
            yield [A, P, N], Y
                
        i = j
        j += batch_size
        count += 1


def build_start_dataset():
    
    #build data frame with apn file paths
    combined_APN_list = []
    threshold = 20

    for i, item in enumerate(search_items):
    
        whale_img_list = get_whale_images(item)

        A, P = zip(*create_a_p_permutations(whale_img_list))
        A = list(A)
        P = list(P)

        #amplify frequency of rare items; they will be matched with more negatives and are therefore useful
        if len(whale_img_list) == 2:
            A = A * 5
            P = P * 5
        elif len(whale_img_list) == 3:
                A = A * 2
                P = P * 2
                #    elif len(whale_img_list) in [4,5,6]:
                #        A = A * 2
                #        P = P * 2

        N = create_n_list(item, A)
        
        A = ['/home/vlad/Documents/Whales/train/' + item for item in A]
        P = ['/home/vlad/Documents/Whales/train/' + item for item in P]
        N = ['/home/vlad/Documents/Whales/train/' + item for item in N]

        APN_list = list(zip(A,P,N))
        
        APN_list = select_img_with_threshold(APN_list, threshold = threshold)
        
        for jtem in APN_list:
            combined_APN_list.append(jtem)

    #Shuffle combined APN list
    shuffle(combined_APN_list)
    
    combined_APN_list = pd.DataFrame(combined_APN_list)

    combined_APN_list['label'] = 1

    combined_APN_list.columns = ['anchor','positive','negative','label']

    return combined_APN_list


def rebuild_dataset():
    
    #build data frame with apn file paths
    combined_APN_list = []
    threshold = 20

    for i, item in enumerate(search_items):
    
        whale_img_list = get_whale_images(item)

        A, P = zip(*create_a_p_permutations(whale_img_list))
        A = list(A)
        P = list(P)

        #amplify frequency of rare items; they will be matched with more negatives and are therefore useful
        if len(whale_img_list) == 2:
            A = A * 5
            P = P * 5
        elif len(whale_img_list) == 3:
                A = A * 2
                P = P * 2
                #    elif len(whale_img_list) in [4,5,6]:
                #        A = A * 2
                #        P = P * 2

        N = create_n_list_cycle(item, A)
        
        A = ['/home/vlad/Documents/Whales/train/' + item for item in A]
        P = ['/home/vlad/Documents/Whales/train/' + item for item in P]
        N = ['/home/vlad/Documents/Whales/train/' + item for item in N]

        APN_list = list(zip(A,P,N))
        
        APN_list = select_img_with_threshold(APN_list, threshold = threshold)
        
        for jtem in APN_list:
            combined_APN_list.append(jtem)

    #Shuffle combined APN list
    shuffle(combined_APN_list)
    
    combined_APN_list = pd.DataFrame(combined_APN_list)

    combined_APN_list['label'] = 1

    combined_APN_list.columns = ['anchor','positive','negative','label']

    return combined_APN_list



#MODEL EXECUTION CODE IS BELOW_________________________________________________
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
test_items = search_items


#the same command will be used inside the loop except selection of negatives; also here it is needed to get params
combined_APN_list = build_start_dataset()


#Upload weights if necessary
#whaleModel_fuse.load_weights('/home/vlad/Documents/Whales/whales_model/whaleModel_fuse_test.h5', by_name=True)
#baseModel.load_weights('/home/vlad/Documents/Whales/whales_model/baseModel_test.h5', by_name=True)


#set model training parameters
num_repeats = 5
batch_size = 64
epochs = 1

train_generator = generator_from_df(combined_APN_list, batch_size=batch_size)

#get number of train rows
ntrain = combined_APN_list.shape[0]
#get number of batches
nbatches_train, mod = divmod(ntrain, batch_size)


test_outputs = []


for i in range(num_repeats):
    
        
    whaleModel_fuse.fit_generator(train_generator, steps_per_epoch=nbatches_train,
                                  epochs = epochs, use_multiprocessing = True, workers = 6)
    
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

    closest_negatives = []
    for i in range(len(test_items)):
        img_path = get_img_path(test_items[i], new_whale = False, img_to_store = False)
        closest_negatives.append(select_closest_negatives(img_path, test_items[i], database, whaleModel))
    closest_negatives = dict(zip(test_items, closest_negatives))
    
    combined_APN_list = rebuild_dataset()
    
    closest_items = []
    for i in range(len(test_items)):
        img_path = get_img_path(test_items[i], new_whale = False, img_to_store = False)
        closest_items.append(get_match_index(img_path, test_items[i], database, whaleModel))
    closest_items = pd.DataFrame(closest_items)
    closest_items.plot(kind = 'hist', bins = 100)
    
