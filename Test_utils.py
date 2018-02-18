#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 08:16:18 2018

@author: vlad
"""

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




#SELECT ALL IDS WHERE NUMBER OF CORRESPONDING IMAGES IS EQUAL TO OR EXCEEDS A SET THRESHOLD
#EXCLUDE new_whale
def select_search_items(img_count_threshold = 2):

    filtered_count_dict = {k: v for k, v in count_dict.items() if v['Image'] >= img_count_threshold and k != 'new_whale'}
    search_items = list(filtered_count_dict.keys())
    
    return search_items




#ENCODE IMAGE INTO A PREDICTED VALUE USING WHALEMODEL.PREDICT TO BE PASSED AS AN ARGUMENT
def img_to_encoding(image_path, model):
    
    img_e = image.load_img(image_path, target_size=(224, 224))
    x_train = image.img_to_array(img_e)
    x_train = np.expand_dims(x_train, axis=0)
    x_train = preprocess_input(x_train)
    
    embedding = model.predict(x_train)
    return embedding





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
    
    if min_dist > 0.45:
        print("Not in the database.")
    else:
        print ("it's " + str(identity) + ", the distance is " + str(min_dist))
        
    return min_dist, identity



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




#Pick test items by running a select_search_items function and choosing 10 ids
test_items = select_search_items(img_count_threshold = 2)
negative_items = test_items[0:10]
test_items = test_items[302:312]




#BUILD DATABASE OF KNOWN WHALES USING KNOWN IMAGES
database = {}
for i in range(len(test_items)):
    img_path = get_img_path(test_items[i], new_whale = False, img_to_store = True)
    database[test_items[i]] = img_to_encoding(img_path, whaleModel)


#FOR WHALES KNOWN TO BE IN DATABASE: def test_items(test_list_number = 0):

def test_positive():
    for item in range(len(test_items)):
        test_img_path = get_img_path(test_items[item], new_whale = False, img_to_store = False)
        id_whale(test_img_path, database, whaleModel)


def test_negative():
    for item in range(len(negative_items)):
        test_img_path = get_img_path(negative_items[item], new_whale = False, img_to_store = False)
        id_whale(test_img_path, database, whaleModel)



'''
RESULTS AFTER 17 EPOCHS WITH CUT-OFF IN ID WHALE SET AT 0.45:
    
FOR WHALES THAT ARE KNOWN TO BE IN THE LIST, FROM MODIFIED IMAGES


it's w_53064a6, the distance is 0.29764804 CORRECT
Not in the database. INCORRECT
it's w_53064a6, the distance is 0.2399287 INCORRECT
Not in the database. INCORRECT
Not in the database. INCORRECT
it's w_ff1b64c, the distance is 0.260149 CORRECT
it's w_b4732ef, the distance is 0.43644214 INCORRECT
it's w_71764b4, the distance is 0.2881321 INCORRECT
Not in the database. INCORRECT
it's w_33973bf, the distance is 0.31689042 CORRECT

SUMMARY: 3 OUT OF 10 CORRECT



5 OUT OF 10 IMAGES CORRECTLY IDENTIFIED AND ATTRIBUTED.

FOR WHALES THAT ARE KNOWN TO BE !!!NOT!!! IN THE LIST, FROM MODIFIED IMAGES
Not in the database.
Not in the database.
Not in the database.
Not in the database.
Not in the database.
it's w_ea2385d, the distance is 0.23558696
Not in the database.
Not in the database.
Not in the database.

'''

'''
test_positive()
it's w_c0cfd5b, the distance is 0.29855856 CORRECT
Not in the database.
it's w_3f365f3, the distance is 0.26271233 CORRECT
Not in the database.
it's w_1632307, the distance is 0.3655736 CORRECT
Not in the database.
it's w_a21cc97, the distance is 0.056676168 INCORRECT
Not in the database.
it's w_a21cc97, the distance is 0.19906016 CORRECT
Not in the database.

test_negative()
Not in the database.
Not in the database.
Not in the database.
Not in the database.
Not in the database.
Not in the database.
Not in the database.
Not in the database.
Not in the database.
Not in the database.
'''



