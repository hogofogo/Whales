#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 12:18:12 2018

@author: vlad
"""

#DATABASE OF WHALE IMAGES NEEDS TO BE ENHANCED, AND THE IMAGE NAMES SAVED IN whale_dict
#[TBD]
#[TBD]
#[TBD]


import csv
import pandas as pd
from PIL import Image
import random



#BUILD WHALE/IMAGE DICTIONARIES Specify the file to be opened, and use the ‘rb’ method meaning “read binary”
reader = csv.DictReader(open('/Users/vlad/Projects/whales/train.csv', 'rb'))
whale_dict = []
for line in reader:
    whale_dict.append(line)
count_dict = pd.pivot_table(pd.DataFrame(whale_dict), index = 'Id', aggfunc = 'count')
whale_dict = {item['Image']:item for item in whale_dict}
count_dict = count_dict.to_dict(orient = 'index')



#SELECT ALL IDS WHERE NUMBER OF CORRESPONDING IMAGES IS EQUAL TO OR EXCEEDS A SET 
#MIN AND MAX THRESHOLDS
#EXCLUDE new_whale
def ds_search_items(img_count_min = 8, img_count_max = 8):

    filtered_count_dict = {k: v for k, v in count_dict.items() if v['Image'] >= img_count_min and v['Image'] <= img_count_max and k != 'new_whale'}
    search_items = list(filtered_count_dict.keys())
    
    return search_items



#CREATE GLOBAL DICTIONARY FOR WHALE REVERSE LOOKUP OF IMAGES BY WHALE ID
search_whale = pd.DataFrame(whale_dict).T


#GET LIST OF ALL ANCHOR FILES (THIS INCLUDES ALL KNOWN WHALES COUNT 2 OR MORE, 
#AND EXCLUDES NEW WHALES)
img_count_threshold = 12
anchor_whale_list = {k: v for k, v in count_dict.items() if v['Image'] >= img_count_threshold and k != 'new_whale'}
#anchor_list WILL ALSO BE USED AS POSITIVE LIST


#SELECT LIST OF IMAGES WITH THRESHOLD >=2 AND EXCLUDE new_whale
search_whale_img_list = search_whale.loc[search_whale['Id'].isin(search_items)] 
search_whale_img_list = search_whale_img_list['Image']
search_whale_img_list = search_whale_img_list.values.tolist()


#FOR EACH SEARCH ITEM, BUILD ANCHOR, POSITIVE AND NEGATIVE DATABASES FOR TRIPLET LOSS EVALUATION
def build_item_APN(search_item, anchors_per_whale = 1, sets_per_anchor = 5):

    '''
    TAKE ONE INDIVIDUAL search_item
    RETURN three numpy arrays of equal length - Anchor, Positive and Negative
    '''
    
    
    #search_whale.set_index(keys = 'Id', inplace = True)
    Pos = search_whale.loc[search_whale['Id'] == search_item]
    Neg = search_whale.loc[search_whale['Id'] != search_item]

    Pos = list(Pos.index.values)
    Neg = list(Neg.index.values)

    #create anchor image from first image for a given item
    
    scroll_list = np.random.choice(Pos, anchors_per_whale, replace = False)
    
    Anchor = []
    Positive = []
    Negative = []
    
    rand_gen = 88
    
    for item in scroll_list:
        
        #the commented out variation of the function below is useful for
        # when instead of random, orderly image needs to be picked up
        #Anchor_array = np.array(Image.open('/Users/vlad/Projects/temp_files/' + Pos[0]),dtype=np.uint8) / 255.
        Anchor_array = np.array(Image.open('/Users/vlad/Projects/temp_files/' + item),dtype=np.uint8) / 255.
        for i in range(sets_per_anchor): 
            Anchor.append(Anchor_array)
            rand_gen += 10
            np.random.seed(rand_gen)
            #the commented out variation of the function below is useful for
            # when instead of random, orderly image needs to be picked up
#            Positive.append(np.array(Image.open('/Users/vlad/Projects/temp_files/' + Pos[i+1]),dtype=np.uint8) / 255.)
            Positive.append(np.array(Image.open('/Users/vlad/Projects/temp_files/' + random.choice(Pos)),dtype=np.uint8) / 255.)
            Negative.append(np.array(Image.open('/Users/vlad/Projects/temp_files/' + random.choice(Neg)),dtype=np.uint8) / 255.)


    return Anchor, Positive, Negative


#BUILD ANCHOR, POSITIVE AND NEGATIVE DATABASES FOR ALL SEARCH ITEMS
def build_all_APN(ds_search_items, anchors_per_whale = 2, sets_per_anchor = 5):
    
    A, P, N = build_item_APN(ds_search_items[0], anchors_per_whale, sets_per_anchor)
    
    for i in range(len(ds_search_items)-1):
        a, p, n = build_item_APN(search_items[i+1])
        A.extend(a)
        P.extend(p)
        N.extend(n)
        print(i)
        
    return A, P, N


#PERFORM DS SEARCH FOR 8 GRAMS
search_items = ds_search_items(img_count_min = 14, img_count_max = 16)
#Anchor, Positive, Negative = build_all_APN(search_items, anchors_per_whale = 2, sets_per_anchor = 5)

#CONVERT RESULTING IMAGE LISTS INTO NUMPY ARRAYS
Anchor = np.array(Anchor)
Positive = np.array(Positive)
Negative = np.array(Negative)

#BUILD A LARGE DATABASE WITH H5PY
import h5py

#write datasets
def Dataset_write(ds_name):
    
    hf = h5py.File('Anchor.h5', 'r+')
    hf.create_dataset(ds_name, data=Anchor)
    hf.close()
    
    hf = h5py.File('Positive.h5', 'r+')
    hf.create_dataset(ds_name, data=Positive)
    hf.close()
    
    hf = h5py.File('Negative.h5', 'r+')
    hf.create_dataset(ds_name, data=Negative)
    hf.close()
    
    return none
    
    
#Dataset_write(ds_name = 'ds1')  -- for 13-34; 10 per whale
#Dataset_write(ds_name = 'ds2')  -- for 9-12; 10 per whale
#Dataset_write(ds_name = 'ds3')  -- for 8; 10 per whale
#Dataset_write(ds_name = 'ds4')  -- for 7; 10 per whale   
#Dataset_write(ds_name = 'ds5')  -- for 6; 10 per whale
#Dataset_write(ds_name = 'ds6')  -- for 5; 5 per whale
#Dataset_write(ds_name = 'ds7')  -- for 4; from list 0 to 100, 3 per whale
#Dataset_write(ds_name = 'ds8')  -- for 4; from list 100 to end, 3 per whale    
#Dataset_write(ds_name = 'ds9')  -- for 3; from list 0 to 250, 2 per whale 
#Dataset_write(ds_name = 'ds10')  -- for 3; from list 250 to end, 2 per whale 
    
#read datasets
hf = h5py.File('Anchor.h5', 'r')
Anchor = hf.get('ds1')
Anchor = np.array(Anchor)
hf.close()


#results after 1 epoch 10 datasets loss 3.2853
#results after 2 epoch 10 datasets loss 2.6648
#results after 3 epoch 10 datasets loss 2.1837
#results after 4 epoch 10 datasets loss 1.9676
#results after 5 epoch 10 datasets loss 1.7172
#results after 6 epoch 10 datasets loss 1.3576
#results after 7 epoch 10 datasets loss 1.4581
#results after 8 epoch 10 datasets loss 1.1587
#results after 9 epoch 10 datasets loss 0.9686
#results after 10 epoch 10 datasets loss 0.7977
#results after 11 epoch 10 datasets loss 0.6936
#results after 12 epoch 10 datasets loss 0.7945
#results after 13 epoch 10 datasets loss 0.6070
#results after 14 epoch 10 datasets loss 0.4241 
#results after 15 epoch 10 datasets loss 0.4435
#results after 16 epoch 10 datasets loss 0.4241
#results after 17 epoch 10 datasets loss 0.4435
#results after 18 epoch 10 datasets loss 0.3266
#results after 19 epoch 10 datasets loss 0.4422
#results after 20 epoch 10 datasets loss 0.2808
#results after 21 epoch 10 datasets loss 0.2586






#whaleModel.save('/Users/vlad/Projects/whales/whales_model/whaleModel.h5')