#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  3 20:09:28 2018

@author: vlad
"""

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
os.chdir('/Users/vlad/Projects/whales')
import csv
import pandas as pd
import keras.preprocessing.image as kim
import re 
import random



#RESIZE ALL FILES AND SAVE THEM TO DIRECTORY 'temp_files'
idealWidth = 224
idealHeight = 224
for f in os.listdir("/Users/vlad/Projects/whales/train"):
    if not f.startswith('.'):
        im = Image.open('/Users/vlad/Projects/whales/train/' + f)
        #resize
        convertedImage = im.resize((idealWidth,idealHeight),Image.NEAREST)
        # convert to greyscale
        convertedImage = convertedImage.convert("LA")
        convertedImage = convertedImage.convert('RGB')
        #save converted files
        saved_location = "/Users/vlad/Projects/temp_files/" + f
        convertedImage.save(saved_location)


#BUILD WHALE/IMAGE DICTIONARIES Specify the file to be opened, and use the ‘rb’ method meaning “read binary”
reader = csv.DictReader(open('/Users/vlad/Projects/whales/train.csv', 'rb'))
whale_dict = []
for line in reader:
    whale_dict.append(line)
count_dict = pd.pivot_table(pd.DataFrame(whale_dict), index = 'Id', aggfunc = 'count')
whale_dict = {item['Image']:item for item in whale_dict}
count_dict = count_dict.to_dict(orient = 'index')





#GET LIST OF ALL FILES IN THE DIRECTORY
files_list = os.listdir("/Users/vlad/Projects/temp_files")
files_list = files_list[1:]

#GET LIST OF ALL ANCHOR FILES (THIS INCLUDES ALL KNOWN WHALES COUNT 2 OR MORE, 
#AND EXCLUDES NEW WHALES)
img_count_threshold = 2
anchor_whale_list = {k: v for k, v in count_dict.items() if v['Image'] >= img_count_threshold and k != 'new_whale'}
#anchor_list WILL ALSO BE USED AS POSITIVE LIST

#CREATE GLOBAL DICTIONARY FOR WHALE REVERSE LOOKUP OF IMAGES BY WHALE ID
search_whale = pd.DataFrame(whale_dict).T


#SELECT LIST OF IMAGES WITH THRESHOLD >=2 AND EXCLUDE new_whale
search_whale_img_list = search_whale.loc[search_whale['Id'].isin(anchor_whale_list)] 
search_whale_img_list = search_whale_img_list['Image']
search_whale_img_list = search_whale_img_list.values.tolist()



#RESIZE IMAGE TO THE DESIRED DIMENSIONS
def resize_img(img):
    
    idealWidth = 224
    idealHeight = 224
    convertedImage = img.resize((idealWidth,idealHeight),Image.NEAREST)
    # convert to greyscale and restore 3 channels
    convertedImage = convertedImage.convert("LA")
    convertedImage = convertedImage.convert('RGB')
    
    return convertedImage




#THIS FUNCTION CREATES A DATA FEED FOR THE MODEL; IT SCROLLS THROUGH ALL THE
#FILES CORRESPONDING TO 2 OR MORE WHALES, AND FEEDS A RANDOM POSITIVE AND A
#RANDOM NEGATIVE IMAGE IN THE FORM OF NUMPY ARRAY
def generate_arrays_from_files():
    while True:
        rand_gen = 0
        for item in search_whale_img_list:
            rand_gen += 1
            np.random.seed(rand_gen)
            # create Numpy arrays of input data
            # anchor, positive and negative
            img_a = Image.open('/Users/vlad/Projects/whales/train/' + item)
            img_a = resize_img(img_a)
            Anchor_array = np.array(img_a,dtype=np.uint8) / 255.
            Anchor_array = np.expand_dims(Anchor_array, axis = 0)
                
            whale_name = search_whale.loc[search_whale['Image'] == item, 'Id'][0]             
            Pos = search_whale.loc[(search_whale['Id'] == whale_name) & (search_whale['Image'] != item)]
            Neg = search_whale.loc[search_whale['Id'] != whale_name]
            
            Pos = list(Pos.index.values)
            Neg = list(Neg.index.values)
            
            img_p = Image.open('/Users/vlad/Projects/whales/train/' + random.choice(Pos))
            img_p = resize_img(img_p)
            #!!!add image manipulation function
            Positive_array = np.array(img_p,dtype=np.uint8) / 255.
            Positive_array = np.expand_dims(Positive_array, axis = 0)
            
            img_n = Image.open('/Users/vlad/Projects/whales/train/' + random.choice(Neg))
            img_n = resize_img(img_n)
            Negative_array = np.array(img_n,dtype=np.uint8) / 255.
            Negative_array = np.expand_dims(Negative_array, axis = 0)
            
        
            y = 1
        
        yield (np.concatenate((Anchor_array, Positive_array, Negative_array), axis = 0), y)     
        #yield ([Anchor_array, Positive_array, Negative_array], y)    
        
        
#!!! TRY ALTERNATIVE VERSION OF THE SAME FUNCTION    
def generate_arrays_from_files():
    while True:
        rand_gen = 0
        for item in search_whale_img_list:
            rand_gen += 1
            np.random.seed(rand_gen)
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
        
            y = 1
        
#        yield (np.concatenate((Anchor_array, Positive_array, Negative_array), axis = 0), y)     
        
        yield ([Anchor_array, Positive_array, Negative_array], y)             

model.fit_generator(generate_arrays_from_files(),
                    steps_per_epoch=1000, epochs=10)








im = Image.open('/Users/vlad/Projects/whales/train/5233abbc.jpg')
width, height = im.size


width =[]
height = []
ratio = []

for f in os.listdir("/Users/vlad/Projects/whales/train"):
    if not f.startswith('.'):
        im = Image.open('/Users/vlad/Projects/whales/train/' + f)
        w, h = im.size
        r = float(w)/float(h)
        ratio.append(r)
        width.append(w)
        height.append(h)
        

plt.hist(width, bins = 50)
show.plt()

plt.hist(height, bins = 50)
show.plt()

plt.hist(ratio, bins = 30)
show.plt()

'''
f25d9acd.jpg
f5a77e06.jpg
e1a6cb3e.jpg
dee277d6.jpg
9ace942b.jpg
74e32bae.jpg
'''


idealWidth = 905
idealHeight = 450
im = Image.open('/Users/vlad/Projects/whales/train/74e32bae.jpg')
resizedChosenImage = im.resize((idealWidth,idealHeight),Image.NEAREST)
plt.imshow(resizedChosenImage)

#suppress background

img = cv2.imread('/Users/vlad/Projects/whales/train/74e32bae.jpg',0)
fgbg = cv2.createBackgroundSubtractorMOG2()
#fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()
fgmask = fgbg.apply(img)

cv2.startWindowThread()
cv2.imshow('frame',fgmask)
cv2.waitKey(5000)

plt.imshow(fgmask)

from skimage import filters
sobel = filters.sobel(img)

plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'
plt.rcParams['figure.dpi'] = 200



# convert all files to grayscale


im = Image.open('/Users/vlad/Projects/whales/train/dee277d6.jpg')
resizedChosenImage = im.resize((idealWidth,idealHeight),Image.NEAREST)
plt.imshow(resizedChosenImage)
greyChosenImage = resizedChosenImage.convert("LA")
plt.imshow(greyChosenImage)



#temp = 'dee277d6.jpg'
#whale_dict[temp]['Id']
#count_dict[temp1]['Image']

# call the number of whale images corresponding to jpeg image
count_dict[whale_dict[temp]['Id']]['Image']

#derive image
def derive_images(imageArray, numRotations):
    #take in imageArray, for each passed image create random rotations = numRotations
    rotationSize = 15
    rotatedImages = [
    kim.random_rotation(imageArray,rotationSize, 
                        row_axis=0, col_axis=1, channel_axis=2, fill_mode='nearest')
    for _ in range(numRotations)]       

    # create random shift images for each of images in rotatedImages
    numShifts = 2
    widthRange = 0.1
    heightRange = 0.2
    shiftedImages = [[
            kim.random_shift(n, wrg= widthRange, hrg= heightRange, 
                     row_axis=0, col_axis=1, channel_axis=2, fill_mode='nearest')  for n in rotatedImages]
            for _ in range(numShifts)]
    # remove one extra dimension
    shiftedImages = np.array(shiftedImages).reshape(-1, *np.array(shiftedImages).shape[-3:])

    # create random zoom images for each of images in shiftedImages
    numZooms = 2
    # !!!!CHECK ZOOM PARAMETERS
    zoomRangeWidth = .7
    zoomRangeHeight = 1.1
    zoomedImages = [[
            kim.random_zoom(n, zoom_range=(zoomRangeWidth,zoomRangeHeight),
                row_axis=0, col_axis=1, channel_axis=2, fill_mode='nearest') for n in shiftedImages]
            for _ in range(numZooms)]
    # remove one extra dimension
    zoomedImages = np.array(zoomedImages).reshape(-1, *np.array(zoomedImages).shape[-3:])

    return zoomedImages



count = 0
for f in os.listdir("/Users/vlad/Projects/temp_files"):
    if (not f.startswith('.') and whale_dict[f]['Id'] != 'new_whale'):
        im = Image.open('/Users/vlad/Projects/temp_files/' + f)
        imageArray = np.array(im)
        if count_dict[whale_dict[f]['Id']]['Image'] == 1:
            numRotations = 4
        elif count_dict[whale_dict[f]['Id']]['Image'] == 2:
            numRotations = 4
        elif count_dict[whale_dict[f]['Id']]['Image'] == 3:
            numRotations = 3
        elif count_dict[whale_dict[f]['Id']]['Image'] in [4,5,6,7]:
            numRotations = 2
        else:
            numRotations = 1
        
        image_output = derive_images(imageArray, numRotations)

        #image save
        c = 0
        for img in image_output:
            c += 1
            img_to_save = Image.fromarray(img, 'RGB')
            new_f = '1' + re.sub(r'^0*', r'', f)
            saved_location = "/Users/vlad/Projects/temp_transformed/" + re.sub(r'(.{4})\s*$', r'', f) + '_' + str(c) + '.jpg'
            img_to_save.save(saved_location)

        count += 1
        if count % 10 == 0: print(count)


#whale_dict[temp]['Id'] != 'new_whale'
#036b9b64.jpg
#        start 5:43 with 50 (count not zeroed out before the run)


#!!! TEMPORARY USEFUL STUFF
temp1 = np.array(shiftedImages).reshape(-1, *np.array(shiftedImages).shape[-3:])
new_arr = arr.reshape(-1, *arr.shape[-2:])
new_arr.shape
# (12, 5, 6)

shiftedImages = [
    kim.random_shift([n for n in rotatedImages], wrg= widthRange, hrg= heightRange, 
                     row_axis=0, col_axis=1, channel_axis=2, fill_mode='nearest')
    for _ in range(numShifts)]



        
      
rotated_im = kim.random_rotation(imageArray, rotationSize, 
                        row_axis=0, col_axis=1, channel_axis=2, fill_mode='nearest')
plt.imshow(rotated_im)            


for key, value in whale_dict.items():
    print value.keys()[0]
    
#use this whale index for sample; 10 images
search_item = 'w_0a97a25'
count_dict['w_0a97a25']

search_whale = pd.DataFrame(whale_dict).T
#search_whale.set_index(keys = 'Id', inplace = True)
Pos = search_whale.loc[search_whale['Id'] == search_item]
Neg = search_whale.loc[search_whale['Id'] == 'new_whale']

Pos = list(Pos.index.values)
Neg = list(Neg.index.values)

#BUILD DATABASE FOR TRIPLE LOSS EVALUATION
l = len(Pos) - 1
#create anchor image
Anchor = np.array(Image.open('/Users/vlad/Projects/temp_files/' + Pos[0]),dtype=np.uint8) / 255.
#create positive images counting number of images for searched item minus 1 (reserve for anchor)
#create negative images counting number of images for searched item minus 1 (reserve for anchor), randomly selected from new_whale
Positive = []
Negative = []
for i in range(len(Pos)-1): 
    Positive.append(np.array(Image.open('/Users/vlad/Projects/temp_files/' + Pos[i+1]),dtype=np.uint8) / 255.)
    Negative.append(np.array(Image.open('/Users/vlad/Projects/temp_files/' + random.choice(Neg)),dtype=np.uint8) / 255.)

Anchor = np.repeat(np.expand_dims(Anchor,axis = 0), len(Pos) - 1, axis = 0)

X_train = np.concatenate((Anchor, Positive, Negative))
y_train = np.concatenate((np.ones(len(Positive)*2), np.zeros(len(Negative))))
y_train = y_train.reshape((len(y_train), 1))
print ("number of training examples = " + str(X_train.shape[0]))
print ("X_train shape: " + str(X_train.shape))
print ("y_train shape: " + str(y_train.shape))



#SELECT ALL IDS WHERE NUMBER OF CORRESPONDING IMAGES IS EQUAL TO OR EXCEEDS A SET THRESHOLD
#EXCLUDE new_whale
def select_search_items(img_count_threshold = 8):

    filtered_count_dict = {k: v for k, v in count_dict.items() if v['Image'] >= img_count_threshold and k != 'new_whale'}
    search_items = list(filtered_count_dict.keys())
    
    return search_items



#FOR EACH SEARCH ITEM, BUILD ANCHOR, POSITIVE AND NEGATIVE DATABASES FOR TRIPLE LOSS EVALUATION
def build_item_APN(search_item):

    '''
    TAKE ONE INDIVIDUAL search_item
    RETURN three numpy arrays of equal length - Anchor, Positive and Negative
    '''
    
    search_whale = pd.DataFrame(whale_dict).T
    #search_whale.set_index(keys = 'Id', inplace = True)
    Pos = search_whale.loc[search_whale['Id'] == search_item]
    Neg = search_whale.loc[search_whale['Id'] != search_item]

    Pos = list(Pos.index.values)
    Neg = list(Neg.index.values)

    #create anchor image from first image for a given item
    
    Anchor = []
    Anchor_array = np.array(Image.open('/Users/vlad/Projects/temp_files/' + Pos[0]),dtype=np.uint8) / 255.
    
    Positive = []
    Negative = []
    
    rand_gen = 1
    
    for i in range(len(Pos)-1): 
        rand_gen += 1
        np.random.seed(rand_gen)
        Anchor.append(Anchor_array)
        Positive.append(np.array(Image.open('/Users/vlad/Projects/temp_files/' + Pos[i+1]),dtype=np.uint8) / 255.)
        Negative.append(np.array(Image.open('/Users/vlad/Projects/temp_files/' + random.choice(Neg)),dtype=np.uint8) / 255.)


    return Anchor, Positive, Negative



#BUILD ANCHOR, POSITIVE AND NEGATIVE DATABASES FOR ALL SEARCH ITEMS
def build_all_APN(search_items):
    
    A, P, N = build_item_APN(search_items[0])
    
    for i in range(len(search_items)-1):
        a, p, n = build_item_APN(search_items[i+1])
        A.extend(a)
        P.extend(p)
        N.extend(n)
        print(i)
        
    return A, P, N



#BUILD APN DATABASE BY SELECTING WHALE IDS WITH VALUES OF THRESHOLD >=20
search_items = select_search_items(img_count_threshold = 20)
Anchor, Positive, Negative = build_all_APN(search_items)

Anchor = np.array(Anchor)
Positive = np.array(Positive)
Negative = np.array(Negative)




#FUNCTION TO GET IMAGE PATH DEPENDING ON TYPE OF IMAGE
def get_img_path(test_item, new_whale = False, img_to_store = False):

    '''
    TAKE ONE INDIVIDUAL test_item, new_whale - whether a whale is idenfied as know in train
    database; img_to_store - whether this particular image will be made part of the lookup database
    RETURN image path 
    '''  
    search_whale = pd.DataFrame(whale_dict).T
    #search_whale.set_index(keys = 'Id', inplace = True)
    if new_whale == False and img_to_store == True:
        img_list = search_whale.loc[search_whale['Id'] == test_item]
        img_list = list(img_list.index.values)
        img_path = '/Users/vlad/Projects/temp_files/' + img_list[0]
    elif new_whale == False and img_to_store == False:
        img_list = search_whale.loc[search_whale['Id'] == test_item]
        img_list = list(img_list.index.values)
        img_path = '/Users/vlad/Projects/temp_files/' + img_list[1]
    else:    
        img_list = search_whale.loc[search_whale['Id'] == 'new_whale']
        img_list = list(img_list.index.values)
        img_path = '/Users/vlad/Projects/temp_files/' + random.choice(img_list)

    return img_path




#ENCODE IMAGE INTO A PREDICTED VALUE USING WHALEMODEL.PREDICT TO BE PASSED AS AN ARGUMENT
def img_to_encoding(image_path, model):
    x_train = np.array(Image.open(image_path),dtype=np.uint8) / 255.
    x_train = np.expand_dims(x_train, axis = 0)
    embedding = model.predict(x_train)
    return embedding




#BUILD DATABASE OF KNOWN WHALES USING KNOWN IMAGES
database = {}
for i in range(len(test_items)):
    img_path = get_img_path(test_items[i], new_whale = False, img_to_store = True)
    database[test_items[i]] = img_to_encoding(img_path, whaleModel)


#Pick test items by running a select_search_items function and choosing 10 ids
test_items = select_search_items(img_count_threshold = 5)
test_items = test_items[302:312]


def id_whale(image_path, database, model):
    """
    Implements whale recognition by finding whether the image is of a known whale
    
    
    Arguments:
    image_path -- path to an image
    database -- database containing image encodings along with the name of the person on the image
    model -- your Inception model instance in Keras
    
    Returns:
    min_dist -- the minimum distance between image_path encoding and the encodings from the database
    identity -- string, the name prediction for the person on image_path
    """
    
    ## Step 1: Compute the target encoding for the image. Use img_to_encoding() see example above. ## (≈ 1 line)
    encoding = img_to_encoding(image_path, model)
    
    ## Step 2: Find the closest encoding ##
    
    # Initialize "min_dist" to a large value, say 100 (≈1 line)
    min_dist = 100
    
    # Loop over the database dictionary's names and encodings.
    for (name, db_enc) in database.items():
        
        # Compute L2 distance between the target "encoding" and the current "emb" from the database. (≈ 1 line)
        dist = np.linalg.norm(encoding - db_enc)

        # If this distance is less than the min_dist, then set min_dist to dist, and identity to name. (≈ 3 lines)
        if dist < min_dist:
            min_dist = dist
            identity = name
    
    if min_dist > 1.3:
        print("Not in the database.")
    else:
        print ("it's " + str(identity) + ", the distance is " + str(min_dist))
        
    return min_dist, identity


#def test_items(test_list_number = 0):
test_img_path = get_img_path(test_items[test_list_number], new_whale = False, img_to_store = False)
id_whale(test_img_path, database, whaleModel)
# after 8 correct 0,3,8,9
# after 9 correct 3,6,9

test_img_path = get_img_path(test_items[9], new_whale = False, img_to_store = False)
id_whale(test_img_path, database, whaleModel)


ip1 = get_img_path(test_items[9], new_whale = False, img_to_store = True)
ip2 = get_img_path(test_items[9], new_whale = False, img_to_store = False)
im = Image.open(ip1)
plt.imshow(im)

plt.imshow(Image.open(get_img_path(test_items[7], new_whale = False, img_to_store = False)))




