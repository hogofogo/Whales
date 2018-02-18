#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 18 13:30:32 2018

@author: vlad
"""

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
y_train = np.ones(len(Anchor)).reshape(len(Anchor),1)
whaleModel_fuse.compile(optimizer = "Adam", loss = triplet_loss)


#Step 3: train the model
whaleModel_fuse.fit(x = [Anchor, Positive, Negative], y = y_train, epochs = 1)
