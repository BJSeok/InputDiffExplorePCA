# train_distinguisher.py
#
# Seoul National University of Science and Technology (SeoulTech)
# Cryptography and Information Security Lab.
# Author: Byoungjin Seok (bjseok@seoultech.ac.kr)
#
# This implementationd was developed based on A. Gohr's implementation (train_nets.py)

import numpy as np
import tensorflow.keras as keras
import pandas as pd
import os

from pickle import dump
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Conv1D, Input, Reshape, Permute, Add, Flatten, BatchNormalization, Activation, concatenate
from tensorflow.keras import backend as K
from tensorflow.keras.regularizers import l2
from datetime import datetime

dirpath = './trained_results/'
crypto_descriptor = {
    "speck_32_64" : {
        "block_size" : 32,
        "word_size" : 16   
    },
    "speck_64_128" : {
        "block_size" : 64,
        "word_size" : 32
    }
}

def cyclic_lr(num_epochs, high_lr, low_lr):
    """Define cyclical learning rates

    Parameters
    ----------
    num_epochs : int
        number of epochs
    high_lr : float
    low_lr : float

    Returns
    -------
    res : float
        Cylical learning rates

    """
    res = lambda i: low_lr + ((num_epochs-1) - i % num_epochs)/(num_epochs-1) * (high_lr - low_lr)

    return(res)

def make_checkpoint(filepath):
    """Make checkpoint

    Parameters
    ----------
    filepath : string
        Filepath of checkpoint file

    Returns
    -------
    res : class
        ModelCheckpoint class

    """
    res = ModelCheckpoint(filepath, monitor='val_loss', save_best_only = True)

    return(res)

def prediction_head(in_layer, regularizer=0.0001):
    """End of the convolutional blocks tower

    Parameters
    ----------
    in_layer : class
        Tensor class

    Returns
    -------
    out : class
        Neural network model

    """
    #add prediction head
    flat1 = Flatten()(in_layer)
    dense1 = Dense(64,kernel_regularizer=l2(regularizer))(flat1)
    dense1 = BatchNormalization()(dense1)
    dense1 = Activation('relu')(dense1)
    dense2 = Dense(64, kernel_regularizer=l2(regularizer))(dense1)
    dense2 = BatchNormalization()(dense2)
    dense2 = Activation('relu')(dense2)
    
    out = Dense(1, activation='sigmoid', kernel_regularizer=l2(regularizer))(dense2)

    return out

def df2ndarray(dataframe):
    """Dataframe to ndarray

    Parameters
    ----------
    dataframe : dataframe
        dataframe of dataset

    Returns
    -------
    ndarr : ndarray
        ndarray of dataset

    """
    ndarr = np.uint8(dataframe.to_numpy())
    return ndarr

def df2TrainValidSet(df1, df2):
    """Dataframe to train set and validation set

    Parameters
    ----------
    df1 : dataframe
        dataframe of dataset ( label : 0 )
    df2 : dataframe
        dataframe of dataset ( label : 1 )

    Returns
    -------
    train_set : dataframe
        dataframe of train set
    valid_set : dataframe
        dataframe of validation set

    """
    concated_df = pd.concat([df1, df2])
    concated_df = concated_df.sample(frac=1).reset_index(drop=True)
    train_set = concated_df.sample(frac=0.7)
    valid_set = concated_df.drop(train_set.index)

    return (train_set, valid_set)

def make_resnet(num_blocks, block_size, word_size, num_filters, kernel_size, depth, regularizer):
    """Make residual tower of convolutional blocks

    Parameters
    ----------
    num_blocks : int
        Number of cipher blocks
    block_size : int
        Cipher block size
    word_size : int
        Cipher word size
    num_filters : int
        Number of filters
    kernel_size : int
        Kernel size
    depth : int
        Depth
    regularizer : float
        Kernel Regularizer

    Returns
    -------
    model : class
        Resnet model

    """
    #Input and preprocessing layers
    inp = Input(shape=(num_blocks * block_size,))
    rs = Reshape((int(block_size / word_size) * num_blocks, word_size))(inp)
    perm = Permute((2,1))(rs)
    
    #add a single residual layer that will expand the data to num_filters channels
    #this is a bit-sliced layer
    conv0 = Conv1D(num_filters, kernel_size=1, padding='same', kernel_regularizer=l2(regularizer))(perm)
    conv0 = BatchNormalization()(conv0)
    conv0 = Activation('relu')(conv0)
    
    #add residual blocks
    shortcut = conv0
    for i in range(depth):
        conv1 = Conv1D(num_filters, kernel_size = kernel_size, padding='same', kernel_regularizer=l2(regularizer))(shortcut)
        conv1 = BatchNormalization()(conv1)
        conv1 = Activation('relu')(conv1)
        conv2 = Conv1D(num_filters, kernel_size = kernel_size, padding='same',kernel_regularizer=l2(regularizer))(conv1)
        conv2 = BatchNormalization()(conv2)
        conv2 = Activation('relu')(conv2)
        shortcut = Add()([shortcut, conv2])
  
    out = prediction_head(shortcut)
    model = Model(inputs=inp, outputs=out)

    return(model)

def train_resnet_distinguisher(X, Y, X_eval, Y_eval, algo, num_rounds, batch_size, num_filters, kernel_size, num_epochs, depth, regularizer):
    """Resnet model training

    Parameters
    ----------
    X : ndarray
        Train dataset
    Y : ndarray
        Train label
    X_eval : ndarray
        Validation dataset
    Y_eval : int
        Validation label
    algo : int
        Algorithm name
    num_rounds : int
        Number of block cipher rounds
    batch_size : float
        Batch size
    num_filters : int
        Number of filters
    kernel_size : int
        Kernel size
    num_epochs : int
        Number of epochs
    depth : int
        Depth
    regularizer : float
        Kernel Regularizer

    Returns
    -------
    model : class
        Resnet model
    h : class
        Resnet history

    """
    num_blocks = int(X.shape[1] / crypto_descriptor[algo]["block_size"])
    block_size = crypto_descriptor[algo]["block_size"]
    word_size = crypto_descriptor[algo]["word_size"]
    
    #create the network
    net = make_resnet(num_blocks, block_size, word_size, num_filters, kernel_size, depth, regularizer)
    net.compile(optimizer='adam',loss='mse',metrics=['acc'])
    net.summary()
    
    #set up model checkpoint
    check = make_checkpoint(dirpath+'best_resnet'+str(num_rounds)+'depth'+str(depth)+'.h5')
    
    #create learnrate schedule
    lr = LearningRateScheduler(cyclic_lr(10, 0.002, 0.0001))
    
    try:
        if not(os.path.isdir(dirpath)):
            os.makedirs(os.path.join(dirpath))
    except OSError as e:
        if e.errno != errno.EXIST:
            print("Failed to create directory!!!!!")
        raise
    
    #train and evaluate
    h = net.fit(X, Y, epochs = num_epochs, batch_size = batch_size, validation_data = (X_eval, Y_eval), callbacks=[lr,check])
    np.save(dirpath+'resnet_'+str(num_rounds)+'r_depth'+str(depth)+"_generated_time_" + datetime.today().strftime("%Y%m%d%H%M%S")+'.npy', h.history['val_acc'])
    np.save(dirpath+'resnet_'+str(num_rounds)+'r_depth'+str(depth)+"_generated_time_" + datetime.today().strftime("%Y%m%d%H%M%S")+'.npy', h.history['val_loss'])
    dump(h.history,open(dirpath+'resnet_hist_'+str(num_rounds)+'r_depth'+str(depth)+"_generated_time_" + datetime.today().strftime("%Y%m%d%H%M%S")+'.p','wb'))

    print("Best validation accuracy: ", np.max(h.history['val_acc']))
    
    return(net, h)

def make_fcnn(num_blocks, block_size, word_size, num_layers, activation = 'relu'):
    """Make fully connected tower of convolutional blocks

    Parameters
    ----------
    num_blocks : int
        Number of cipher blocks
    block_size : int
        Cipher block size
    word_size : int
        Cipher word size
    num_layers : int
        Number of layers
    activation : string
        activation function

    Returns
    -------
    model : class
        FCNN model

    """
    model = keras.Sequential([keras.layers.InputLayer(input_shape = (num_blocks * block_size, )),])
    
    for i in range(num_layers):
        model.add(keras.layers.Dense(num_blocks * block_size, activation=activation))
       
    model.add(keras.layers.Dense(1, activation='sigmoid'))
    
    return model

def train_fcnn_distinguisher(X, Y, X_eval, Y_eval, algo, num_rounds, batch_size, num_epochs, num_layers):
    """FCNN model training

    Parameters
    ----------
    X : ndarray
        Train dataset
    Y : ndarray
        Train label
    X_eval : ndarray
        Validation dataset
    Y_eval : int
        Validation label
    algo : int
        Algorithm name
    num_rounds : int
        Number of block cipher rounds
    batch_size : float
        Batch size
    num_epochs : int
        Number of epochs
    num_layers : int
        Number of layers

    Returns
    -------
    model : class
        FCNN model
    h : class
        FCNN history

    """
    num_blocks = int(X.shape[1] / crypto_descriptor[algo]["block_size"])
    block_size = crypto_descriptor[algo]["block_size"]
    word_size = crypto_descriptor[algo]["word_size"]
    
    #create the network
    net = make_fcnn(num_blocks, block_size, word_size, num_layers)
    net.compile(optimizer='adam', loss='mse', metrics=['acc'])
    net.summary()
    
    #set up model checkpoint
    check = make_checkpoint(dirpath+'best_fcnn_r'+str(num_rounds)+'_l'+str(num_layers)+"_generated_time_" + datetime.today().strftime("%Y%m%d%H%M%S") + ".h5")

    #create learnrate schedule
    lr = LearningRateScheduler(cyclic_lr(10, 0.002, 0.0001))

    try:
        if not(os.path.isdir(dirpath)):
            os.makedirs(os.path.join(dirpath))
    except OSError as e:
        if e.errno != errno.EXIST:
            print("Failed to create directory!!!!!")
        raise
    
    h = net.fit(X, Y, epochs = num_epochs, batch_size = batch_size, validation_data=(X_eval, Y_eval), callbacks=[lr, check])
    np.save(dirpath+'dnn_'+str(num_rounds)+'r_layer'+str(num_layers)+"_generated_time_" + datetime.today().strftime("%Y%m%d%H%M%S") + '.npy', h.history['val_acc'])
    np.save(dirpath+'dnn_'+str(num_rounds)+'r_layer'+str(num_layers)+"_generated_time_" + datetime.today().strftime("%Y%m%d%H%M%S") + '.npy', h.history['val_loss'])
    dump(h.history,open(dirpath+'dnn_hist_'+str(num_rounds)+'r_layer'+str(num_layers)+"_generated_time_" + datetime.today().strftime("%Y%m%d%H%M%S") + '.p','wb'))
    
    print("Best validation accuracy: ", np.max(h.history['val_acc'])) 
    
    return (net, h)

def block(x, growth_rate, regularizer):
    """Make block

    Parameters
    ----------
    x : class
        Tensor class
    growth_rate : int
        Growth rate
    regularizer : float
        Kernel Regularizer

    Returns
    -------
    x : class
        Tensor class

    """
    x1 = BatchNormalization()(x)
    x1 = Activation('relu')(x)
    x1 = Conv1D(4*growth_rate, kernel_size=3, padding='same', kernel_regularizer=l2(regularizer))(x)
    x = concatenate([x, x1])
    return x

def dense_layer(x, num_blocks, growth_rate, regularizer):
    """Make dense layer

    Parameters
    ----------
    x : class
        Tensor class
    num_blocks : int
        Number of cipher blocks
    growth_rate : int
        Growth rate
    regularizer : float
        Kernel Regularizer

    Returns
    -------
    x : class
        Tensor class

    """
    for i in range(num_blocks):
        x = block(x, growth_rate, regularizer)
    return x

def transition_layer(x, num_filters, regularizer):
    """Make transition layer

    Parameters
    ----------
    x : class
        Tensor class
    num_filters : int
        Number of filters
    regularizer : float
        Kernel Regularizer

    Returns
    -------
    x : class
        Tensor class

    """
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv1D(num_filters, kernel_size=3, padding='same', kernel_regularizer=l2(regularizer))(x)
    return x

def make_densenet(num_blocks, block_size, word_size, num_filters, kernel_size, depth, growth_rate, regularizer):
    """Make Densenet tower of convolutional blocks

    Parameters
    ----------
    num_blocks : int
        Number of cipher blocks
    block_size : int
        Cipher block size
    word_size : int
        Cipher word size
    num_filters : int
        Number of filters
    kernel_size : int
        Kernel size
    depth : int
        Depth
    growth_rate : int
        Growth rate
    regularizer : float
        Kernel Regularizer

    Returns
    -------
    model : class
        Densenet model

    """
    #Input and preprocessing layers
    inp = Input(shape=(num_blocks * block_size,))
    rs = Reshape((int(block_size / word_size) * num_blocks, word_size))(inp)
    perm = Permute((2,1))(rs)

    #add a single residual layer that will expand the data to num_filters channels
    #this is a bit-sliced layer
    conv0 = Conv1D(num_filters, kernel_size=1, padding='same', kernel_regularizer=l2(regularizer))(perm)
    conv0 = BatchNormalization()(conv0)
    conv0 = Activation('relu')(conv0)
    
    #add dense layers and transition layers
    for i in range(depth):
        conv0 = dense_layer(conv0, num_blocks, growth_rate, regularizer)
        conv0 = transition_layer(conv0, num_filters, regularizer)

    out = prediction_head(conv0)
    model = Model(inputs=inp, outputs=out)
    
    return(model)

def train_densenet_distinguisher(X, Y, X_eval, Y_eval, algo, num_rounds, batch_size, num_filters, kernel_size, num_epochs, depth, growth_rate, regularizer):
    """Densenet model training

    Parameters
    ----------
    X : ndarray
        Train dataset
    Y : ndarray
        Train label
    X_eval : ndarray
        Validation dataset
    Y_eval : int
        Validation label
    algo : int
        Algorithm name
    num_rounds : int
        Number of block cipher rounds
    batch_size : float
        Batch size
    num_filters : int
        Number of filters
    kernel_size : int
        Kernel size
    num_epochs : int
        Number of epochs
    depth : int
        Depth
    growth_rate : int
        Growth rate
    regularizer : float
        Kernel Regularizer

    Returns
    -------
    model : class
        Densenet model
    h : class
        Densenet history

    """
    num_blocks = int(X.shape[1] / crypto_descriptor[algo]["block_size"])
    block_size = crypto_descriptor[algo]["block_size"]
    word_size = crypto_descriptor[algo]["word_size"]

    #create the network
    net = make_densenet(num_blocks, block_size, word_size, num_filters, kernel_size, depth, growth_rate, regularizer)
    net.compile(optimizer='adam',loss='mse',metrics=['acc'])
    net.summary()
    
    #set up model checkpoint
    check = make_checkpoint(dirpath+'best_densenet_r'+str(num_rounds)+'depth'+str(depth)+'.h5')
    
    #create learnrate schedule
    lr = LearningRateScheduler(cyclic_lr(10, 0.002, 0.0001))

    try:
        if not(os.path.isdir(dirpath)):
            os.makedirs(os.path.join(dirpath))
    except OSError as e:
        if e.errno != errno.EXIST:
            print("Failed to create directory!!!!!")
        raise
    
    #train and evaluate
    h = net.fit(X,Y,epochs=num_epochs,batch_size=batch_size,validation_data=(X_eval, Y_eval), callbacks=[lr,check])
    np.save(dirpath+'densenet_'+str(num_rounds)+'r_depth'+str(depth)+"_generated_time_" + datetime.today().strftime("%Y%m%d%H%M%S")+'.npy', h.history['val_acc'])
    np.save(dirpath+'densenet_'+str(num_rounds)+'r_depth'+str(depth)+"_generated_time_" + datetime.today().strftime("%Y%m%d%H%M%S")+'.npy', h.history['val_loss'])
    dump(h.history,open(dirpath+'densenet_hist'+str(num_rounds)+'r_depth'+str(depth)+"_generated_time_" + datetime.today().strftime("%Y%m%d%H%M%S")+'.p','wb'))

    print("Best validation accuracy: ", np.max(h.history['val_acc'])) 

    return(net, h)

