# -*- coding: utf-8 -*-
"""
Created on Thu Aug 13 22:19:12 2015

@author: rc, alex
"""

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D
from keras.layers.recurrent import GRU, LSTM
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import PReLU
from keras.optimizers import SGD, Adam
from keras.regularizers import l2


def buildNN(architecture,training_params,input_dim):
    """Lay out a Neural Network as described in the YAML file."""
    current_units = input_dim
    
    model = Sequential()
    
    for layer in architecture:
        layer_name = layer.keys()[0]
        layer_params = layer[layer_name]
        if layer_name == 'Activation':
            model.add(Activation(layer_params['type']))
        if layer_name == 'PReLU':
            model.add(PReLU(current_units))
        if layer_name == 'Dropout':
            model.add(Dropout(layer_params['p']))
        if layer_name == 'Dense':
            model.add(Dense(current_units, layer_params['num_units'], init='glorot_uniform'))
            current_units = layer_params['num_units']
        if layer_name == 'Conv':
            # a filter covers all predictions for a single time sample
            # strides are made between time samples
            model.add(Convolution2D(layer_params['nb_filters'], 1, current_units, 1,
                                    subsample=(current_units,1),
                                    init='glorot_uniform'))
            current_units = layer_params['nb_filters']*training_params['num_strides']
        if layer_name == 'GRU':
            model.add(GRU(current_units,layer_params['num_units'], return_sequences=layer_params['next_GRU']))
            current_units = layer_params['num_units']
        if layer_name == 'LSTM':
            model.add(LSTM(current_units,layer_params['num_units'], return_sequences=layer_params['next_GRU']))
            current_units = layer_params['num_units']
        if layer_name == 'BatchNormalization':
            model.add(BatchNormalization((current_units,)))
        if layer_name == 'Flatten':
            model.add(Flatten())
        if layer_name == 'Output':
            model.add(Dense(current_units, 6, init='glorot_uniform'))
            model.add(Activation('sigmoid'))
    if not training_params.has_key('optim') or training_params['optim'] == 'sgd':
        optim = SGD(lr=training_params['lr'], decay=float(training_params['decay']), momentum=training_params['momentum'], nesterov=True)    
    elif training_params['optim'] == 'adam':
        optim = Adam(lr=training_params['lr'])
    
    model.compile(loss='binary_crossentropy', optimizer=optim)
    
    return model
    